use std::cmp::Ordering;
use std::io::{self, BufWriter, Write};
use std::net::TcpStream;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::{Duration, Instant};

use crossterm::ExecutableCommand;
use crossterm::event::{self, Event, KeyCode, KeyEvent};
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::prelude::*;

use rand::RngCore;
use ratatui::widgets::{Block, Borders, List, ListItem, Padding, Paragraph, Wrap};
use tfhe::{ClientKey, CompressedFheBool, FheUint16, prelude::*};
use tfhe::ConfigBuilder;
use tfhe_demo::protocol::{SensorMessage, SensorResponse, SetupAck, SetupRequest};
use tfhe_demo::trivium_bool::{TRIVIUM_KEY_BITS, TriviumStream};
use tfhe_demo::utils::{ProgressWriter, bytes_to_bits, encrypt_samples};

use crate::gcm::{GlucoseMonitor, SyntheticGcm};

mod gcm;

pub const SENSOR_INTERVAL_SECS: u64 = 0;
pub const SERVER_ADDR: &str = "127.0.0.1:7879"; //TODO: move to dotenv
pub const DEMO_NUMBER: u8 = 42;

#[derive(Debug)]
enum WorkerMsg {
    Phase(Phase),
    Log(String),
    IterResult { iter: u64, value: [f32; 4], result: u16 },
    Error(String),
}

#[derive(Debug, Clone, PartialEq)]
enum Phase {
    KeyGen,
    FheKeyEncrypt,
    SendSetup,
    ServerSetup,
    Sensing,
    Finished,
}

impl Phase {
    fn label(&self) -> &'static str {
        match self {
            Phase::KeyGen => "Key Generation",
            Phase::FheKeyEncrypt => "FHE Key Encryption",
            Phase::SendSetup => "Key Exchange",
            Phase::ServerSetup => "Server Warmup",
            Phase::Sensing => "Sensor Loop",
            Phase::Finished => "Complete",
        }
    }
    fn icon(&self) -> &'static str {
        match self {
            Phase::KeyGen => "🔑",
            Phase::FheKeyEncrypt => "🛡 ",
            Phase::SendSetup => "📡",
            Phase::ServerSetup => "⚙ ",
            Phase::Sensing => "📶",
            Phase::Finished => "✅",
        }
    }
}

struct App {
    phase: Phase,
    logs: Vec<String>,
    iter_count: u64,
    latest: Option<([f32; 4], u16)>,
    error: Option<String>,
    start: Instant,
    tick: u64,
}

impl App {
    fn new() -> Self {
        Self {
            phase: Phase::KeyGen,
            logs: Vec::new(),
            iter_count: 0,
            latest: None,
            error: None,
            start: Instant::now(),
            tick: 0,        }
    }

    fn apply(&mut self, msg: WorkerMsg) {
        match msg {
            WorkerMsg::Phase(p) => {
                self.logs.push(format!(
                    "[{:.1}s] ▶ {}",
                    self.start.elapsed().as_secs_f64(),
                    p.label()
                ));
                self.phase = p;
            }
            WorkerMsg::Log(s) => {
                self.logs.push(format!(
                    "[{:.1}s] {}",
                    self.start.elapsed().as_secs_f64(),
                    s
                ));
            }
            WorkerMsg::IterResult {
                iter,
                value,
                result,
            } => {
                self.iter_count = iter;
                self.latest = Some((value, result));
                self.logs.push(format!(
                    "[{:.1}s] #{iter}: {:?} → {} ✓",
                    self.start.elapsed().as_secs_f64(),
                    value,
                    result,
                ));
            }
            WorkerMsg::Error(e) => {
                self.error = Some(e);
                self.phase = Phase::Finished;
            }
        }
    }
}

fn main() -> io::Result<()> {
    enable_raw_mode()?;
    io::stdout().execute(EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)?;

    let (tx, rx) = mpsc::channel::<WorkerMsg>();
    thread::spawn(move || run(tx));

    let mut app = App::new();
    let result = run_ui(&mut terminal, &mut app, rx);

    disable_raw_mode()?;
    io::stdout().execute(LeaveAlternateScreen)?;
    result
}

fn run(tx: Sender<WorkerMsg>) {
    macro_rules! log {
        ($($t:tt)*) => { tx.send(WorkerMsg::Log(format!($($t)*))).ok(); };
    }
    macro_rules! phase {
        ($p:expr) => {
            tx.send(WorkerMsg::Phase($p)).ok();
        };
    }
    macro_rules! bail {
        ($($t:tt)*) => {{
            tx.send(WorkerMsg::Error(format!($($t)*))).ok();
            return;
        }};
    }

    phase!(Phase::KeyGen);
    log!("Generating TFHE key pair …");
    let t0 = Instant::now();

    // We use CompressedServerKey to reduce the required bandwidth by ~50%.
    // Experimental results show 150MB -> 50MB transmission sizes.
    // Unfortunately the high entropy renders any further compression by e.g zlib useless.

    let config = ConfigBuilder::default().build();
    let client_key = ClientKey::generate(config);
    let compressed_server_key = client_key.generate_compressed_server_key();
    log!("Keys generated in {:.2}s", t0.elapsed().as_secs_f64());

    let mut trivium_key = [0u8; 10];
    let mut trivium_iv = [0u8; 10];
    rand::thread_rng().fill_bytes(&mut trivium_key);
    rand::thread_rng().fill_bytes(&mut trivium_iv);
    log!("Trivium key : {}", hex(&trivium_key));
    log!("Trivium IV  : {}", hex(&trivium_iv));

    phase!(Phase::FheKeyEncrypt);
    log!("Encrypting {} key bits with TFHE …", TRIVIUM_KEY_BITS);
    let t0 = Instant::now();
    let key_bits = (0..80usize)
        .map(|i| {
            let bit = ((trivium_key[i / 8] >> (i % 8)) & 1) != 0;
            CompressedFheBool::encrypt(bit, &client_key)
        })
        .collect::<Vec<_>>();
    log!(
        "FHE key encryption done in {:.2}s",
        t0.elapsed().as_secs_f64()
    );

    let setup_request = SetupRequest {
        server_key_bytes: bincode::serialize(&compressed_server_key).expect("serialise compresed_server_key"),
        encrypted_key_bits: bincode::serialize(&key_bits).expect("serialise key_bits"),
        iv: trivium_iv,
    };

    log!("Setup payload: {} MB", setup_request.size_of() / 1_048_576);

    // 3. Connect & send SetupRequest ──────────────────────────────────────────
    phase!(Phase::SendSetup);
    log!("Connecting to {} …", SERVER_ADDR);
    let mut conn = match TcpStream::connect(SERVER_ADDR) {
        Ok(c) => c,
        Err(e) => bail!("Connection failed: {e}"),
    };

    log!("Connected. Uploading setup payload …");
    {
        let mut pw = ProgressWriter::new(
            BufWriter::new(&mut conn),
            Duration::from_secs(2),
            |bytes_sent| {
                tx
                .send(WorkerMsg::Log(format!(
                    "Uploading … {:.1} MB sent",
                    bytes_sent as f64 / 1_048_576.0,
                )))
                .ok();
            }
        );
        if let Err(e) = bincode::serialize_into(&mut pw, &setup_request) {
            bail!("Send failed: {e}");
        }
        if let Err(e) = pw.flush() {
            bail!("Send flush failed: {e}");
        }
        log!(
            "Upload complete — {:.1} MB sent.",
            pw.bytes_sent() as f64 / 1_048_576.0
        );
    }

    phase!(Phase::ServerSetup);
    log!("Waiting for server warmup (1152 FHE rounds) …");
    conn.set_read_timeout(Some(Duration::from_secs(3600))).ok();
    if let Err(e) = bincode::deserialize_from::<_, SetupAck>(&mut conn) {
        bail!("Setup handshake failed: {e}");
    }
    log!("Server ready!");

    let key_bool: [bool; 80] = bytes_to_bits(&trivium_key);
    let iv_bool: [bool; 80] = bytes_to_bits(&trivium_iv);
    let mut plain_stream = TriviumStream::<bool>::new(key_bool, iv_bool);

    phase!(Phase::Sensing);
    log!(
        "Entering sensor loop (interval: {}s) …",
        SENSOR_INTERVAL_SECS
    );

    let mut iteration: u64 = 0;

    let cgm: Box<dyn GlucoseMonitor> = Box::new(SyntheticGcm::new());
    let mut samples = std::array::from_fn::<_, 4, _>(|_| cgm.sample());

    loop {
        iteration += 1;

        // Sample the glucose monitor and slide the window.
        samples.rotate_left(1);
        *samples.last_mut().unwrap() = cgm.sample();

        let t0 = Instant::now();
        let enc_samples = encrypt_samples(&mut plain_stream, &samples.map(|sample| sample as u16));
        log!("Samples encrypted in {:.2}ms", t0.elapsed().as_millis());


        let t0 = Instant::now();
        if let Err(e) = bincode::serialize_into(&mut conn, &SensorMessage { trivium_ciphertext: enc_samples.to_vec() }) {
            bail!("Send failed on iteration {iteration}: {e}");
        }
        if let Err(e) = conn.flush() {
            bail!("Flush failed on iteration {iteration}: {e}");
        }
        log!("Samples sent in {:.2}ms", t0.elapsed().as_millis());

        let response = match bincode::deserialize_from::<_, SensorResponse>(&mut conn) {
            Ok(r) => r,
            Err(e) => bail!("Recv failed on iteration {iteration}: {e}"),
        };

        let response: FheUint16 =
            bincode::deserialize(&response.result_bytes[0]).expect("deserialise response");

        let t0 = Instant::now();
        let result = response.decrypt(&client_key);
        log!("Samples decrypted in {:.2}ms", t0.elapsed().as_millis());


        tx.send(WorkerMsg::IterResult {
            iter: iteration,
            value: samples,
            result,
        })
        .ok();

        thread::sleep(Duration::from_secs(SENSOR_INTERVAL_SECS));
    }
}

fn run_ui(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
    rx: Receiver<WorkerMsg>,
) -> io::Result<()> {
    loop {
        while let Ok(msg) = rx.try_recv() {
            app.apply(msg);
        }
        app.tick += 1;
        terminal.draw(|f| render(f, app))?;

        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(KeyEvent { code, .. }) = event::read()? {
                if matches!(code, KeyCode::Char('q') | KeyCode::Esc) {
                    break;
                }
            }
        }
    }
    Ok(())
}

fn render(f: &mut Frame, app: &App) {
    let area = f.area();
    let root = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5),
            Constraint::Min(10),
            Constraint::Length(1),
        ])
        .split(area);

    render_banner(f, root[0], app);
    render_body(f, root[1], app);
    render_statusbar(f, root[2], app);
}

fn render_banner(f: &mut Frame, area: Rect, app: &App) {
    let spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    let spin = spinner[(app.tick / 2) as usize % spinner.len()];

    let title_lines = vec![
        Line::from(vec![
            Span::styled(
                "  ████████╗███████╗██╗  ██╗███████╗  ",
                Style::new().fg(Color::Cyan).bold(),
            ),
            Span::styled(
                "  Trivium over FHE — IoT Sensor  ",
                Style::new().fg(Color::White).bold(),
            ),
        ]),
        Line::from(vec![
            Span::styled(
                "     ██╔══╝██╔════╝██║  ██║██╔════╝  ",
                Style::new().fg(Color::Cyan),
            ),
            Span::styled(
                format!("  {} {} ", spin, app.phase.label()),
                Style::new().fg(Color::Yellow).bold(),
            ),
        ]),
        Line::from(vec![
            Span::styled(
                "     ██║   █████╗  ███████║█████╗    ",
                Style::new().fg(Color::Cyan),
            ),
            Span::styled(
                format!("  Elapsed: {:.1}s", app.start.elapsed().as_secs_f64()),
                Style::new().fg(Color::Green),
            ),
        ]),
    ];

    let banner = Paragraph::new(title_lines).block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::new().fg(Color::Cyan))
            .padding(Padding::horizontal(1)),
    );
    f.render_widget(banner, area);
}

fn render_body(f: &mut Frame, area: Rect, app: &App) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
        .split(area);
    render_pipeline(f, cols[0], app);
    render_right_panel(f, cols[1], app);
}

fn render_pipeline(f: &mut Frame, area: Rect, app: &App) {
    let phases = [
        Phase::KeyGen,
        Phase::FheKeyEncrypt,
        Phase::SendSetup,
        Phase::ServerSetup,
        Phase::Sensing,
        Phase::Finished,
    ];

    let current_idx = phases.iter().position(|p| p == &app.phase).unwrap_or(0);
    let items: Vec<ListItem> = phases
        .iter()
        .enumerate()
        .map(|(i, phase)| {
            let (status_style, prefix) = match i.cmp(&current_idx) {
                Ordering::Less => (Style::new().fg(Color::Green), "✓"),
                Ordering::Equal => (Style::new().fg(Color::Yellow).bold(), "▶"),
                Ordering::Greater => (Style::new().fg(Color::DarkGray), "○"),
            };

            let line = Line::from(vec![
                Span::styled(format!(" {prefix} "), status_style),
                Span::styled(phase.icon(), Style::new()),
                Span::styled(format!(" {}", phase.label()), status_style),
            ]);

            ListItem::new(line)
        })
        .collect();

    let block = Block::default()
        .title(" Pipeline ")
        .title_style(Style::new().fg(Color::Cyan).bold())
        .borders(Borders::ALL)
        .border_style(Style::new().fg(Color::Blue))
        .padding(Padding::vertical(1));

    let list = List::new(items).block(block);
    f.render_widget(list, area);
}

fn render_right_panel(f: &mut Frame, area: Rect, app: &App) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
        .split(area);

    render_log(f, rows[0], app);
    render_result(f, rows[1], app);
}

fn render_log(f: &mut Frame, area: Rect, app: &App) {
    let max_lines = (area.height as usize).saturating_sub(2);
    let start = app.logs.len().saturating_sub(max_lines);
    let items: Vec<ListItem> = app.logs[start..]
        .iter()
        .map(|l| {
            let style = if l.contains("Error") || l.contains("fail") {
                Style::new().fg(Color::Red)
            } else if l.contains("▶") {
                Style::new().fg(Color::Yellow).bold()
            } else if l.contains(" ✓") {
                Style::new().fg(Color::Green)
            } else if l.contains("done") || l.contains("generated") || l.contains("ready") {
                Style::new().fg(Color::Green)
            } else {
                Style::new().fg(Color::White)
            };
            ListItem::new(Line::from(Span::styled(l.as_str(), style)))
        })
        .collect();

    let block = Block::default()
        .title(" Event Log ")
        .title_style(Style::new().fg(Color::Cyan).bold())
        .borders(Borders::ALL)
        .border_style(Style::new().fg(Color::Blue));

    f.render_widget(List::new(items).block(block), area);
}

fn render_result(f: &mut Frame, area: Rect, app: &App) {
    let block = Block::default()
        .title(" Sensor Readings ")
        .title_style(Style::new().fg(Color::Cyan).bold())
        .borders(Borders::ALL)
        .border_style(Style::new().fg(Color::Blue))
        .padding(Padding::horizontal(1));

    if let Some(err) = &app.error {
        let text = Paragraph::new(format!("Error: {err}"))
            .style(Style::new().fg(Color::Red))
            .block(block)
            .wrap(Wrap { trim: false });
        f.render_widget(text, area);
        return;
    }

    if let Some((value, result)) = app.latest {
        let lines = vec![
            Line::from(vec![
                Span::styled("  Readings sent : ", Style::new().fg(Color::DarkGray)),
                Span::styled(
                    format!("{}", app.iter_count),
                    Style::new().fg(Color::Cyan).bold(),
                ),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("  Sensor samples  : ", Style::new().fg(Color::DarkGray)),
                Span::styled(format!("{value:?}"), Style::new().fg(Color::White).bold()),
            ]),
            Line::from(vec![
                Span::styled("  Server result : ", Style::new().fg(Color::DarkGray)),
                Span::styled(
                    format!("{result}"),
                    Style::new()
                        .fg(Color::Green)
                        .bold()
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(""),
            Line::from(Span::styled(
                format!("  Next reading in {}s …", SENSOR_INTERVAL_SECS),
                Style::new().fg(Color::DarkGray).italic(),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "  Press 'q' or Esc to quit",
                Style::new().fg(Color::DarkGray).italic(),
            )),
        ];
        let p = Paragraph::new(lines)
            .block(block)
            .wrap(Wrap { trim: false });
        f.render_widget(p, area);
    } else {
        let sensing = app.phase == Phase::Sensing;
        let lines = vec![
            Line::from(Span::styled(
                if sensing {
                    "  Awaiting first reading …"
                } else {
                    "  Waiting for server …"
                },
                Style::new().fg(Color::DarkGray),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "  The server holds the FHE key and",
                Style::new().fg(Color::DarkGray),
            )),
            Line::from(Span::styled(
                "  computes on each sensor reading",
                Style::new().fg(Color::DarkGray),
            )),
            Line::from(Span::styled(
                "  without ever seeing plaintext.",
                Style::new().fg(Color::DarkGray),
            )),
        ];
        let p = Paragraph::new(lines)
            .block(block)
            .wrap(Wrap { trim: false });
        f.render_widget(p, area);
    }
}

fn render_statusbar(f: &mut Frame, area: Rect, app: &App) {
    let status = match (&app.error, &app.phase) {
        (Some(_), _) => Span::styled(
            " [ERROR] Press 'q' to exit ",
            Style::new().fg(Color::Red).bold(),
        ),
        (_, Phase::Sensing) => Span::styled(
            format!(
                " 📶 Sensor loop active — {} reading(s) sent — Press 'q' to quit ",
                app.iter_count
            ),
            Style::new().fg(Color::Cyan).bold(),
        ),
        (_, Phase::Finished) => Span::styled(
            " [DONE] Session ended — Press 'q' to exit ",
            Style::new().fg(Color::Green).bold(),
        ),
        _ => Span::styled(
            " Setting up … Press 'q' to abort ",
            Style::new().fg(Color::DarkGray),
        ),
    };

    let bar = Paragraph::new(Line::from(status)).alignment(Alignment::Left);
    f.render_widget(bar, area);
}

fn hex(data: &[u8]) -> String {
    data.iter().map(|b| format!("{b:02x}")).collect::<String>()
}
