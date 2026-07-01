use std::error::Error;
use std::io::Write;
use std::net::{TcpListener, TcpStream};
use std::time::Instant;
use tfhe::{CompressedFheBool, CompressedServerKey, FheBool};
use tfhe_demo::protocol::{SensorMessage, SensorResponse, SetupAck, SetupRequest};
use tfhe_demo::trivium_bool::{TRIVIUM_KEY_BITS, TriviumStream};
use tfhe_demo::utils::{bytes_to_bits, decrypt_samples};

use crate::model::{GlucoseModel, NaiveLinearRegression};

mod model;

pub const BIND_ADDR: &str = "0.0.0.0:7879";


fn main() {
    let addr = BIND_ADDR;
    let listener = TcpListener::bind(addr).expect("failed to bind");

    println!("╔══════════════════════════════════════════════════╗");
    println!("║         TFHE Trivium Demo — Server               ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!("Listening on {addr}");
    println!();

    for stream in listener.incoming() {
        let mut stream = stream.expect("accept failed");
        println!("[*] Client connected from {}", stream.peer_addr().unwrap());

        match handle_client(&mut stream) {
            Ok(n) => println!("[*] Session ended after {n} sensor readings. Waiting for next client …"),
            Err(e) => eprintln!("[!] Error: {e}"),
        }
        println!();
    }
}

fn handle_client(stream: &mut TcpStream) -> Result<u64, Box<dyn Error>> {

    println!("[1/4] Receiving SetupRequest …");
    let req = bincode::deserialize_from::<_, SetupRequest>(&mut *stream)?;

    println!("[2/4] Deserialising ServerKey + key bits …");
    let t0 = Instant::now();
    let compressed_server_key = bincode::deserialize::<CompressedServerKey>(&req.server_key_bytes)?;
    let server_key = compressed_server_key.decompress();
    println!("      ServerKey ready in {:.2}s", t0.elapsed().as_secs_f64());

    let key_bits: Vec<CompressedFheBool> = bincode::deserialize(&req.encrypted_key_bits)?;
    let key_arr: [FheBool; TRIVIUM_KEY_BITS] = key_bits
        .iter()
        .map(CompressedFheBool::decompress)
        .collect::<Vec<_>>()
        .try_into()
        .map_err(|_| "expected exactly 80 key bits")?;
    let iv_bool: [bool; TRIVIUM_KEY_BITS] = bytes_to_bits(&req.iv);

    println!("[3/4] Initialising Trivium FHE stream (1152 warmup rounds) …");
    let t0 = Instant::now();
    let mut trivium = TriviumStream::<FheBool>::new(key_arr, iv_bool, &server_key);
    println!("      Warmup done in {:.2}s", t0.elapsed().as_secs_f64());

    // The constructor unsets the server key; re-set it for subsequent operations.
    tfhe::set_server_key(server_key.clone());

    println!("[4/4] Sending SetupAck — entering sensor loop …");
    bincode::serialize_into(&mut *stream, &SetupAck)?;
    stream.flush()?;

    let model: Box<dyn GlucoseModel> = Box::new(NaiveLinearRegression);
    let mut iteration: u64 = 0;
    loop {
        let msg = match bincode::deserialize_from::<_, SensorMessage>(&mut *stream) {
            Ok(m) => m,
            Err(e) => {
                // Distinguish clean disconnect from real errors.
                if let bincode::ErrorKind::Io(io_err) = e.as_ref() {
                    if io_err.kind() == std::io::ErrorKind::UnexpectedEof
                        || io_err.kind() == std::io::ErrorKind::ConnectionReset
                    {
                        println!("[*] Client disconnected.");
                        break;
                    }
                }
                return Err(e);
            }
        };

        iteration += 1;
        let t0 = Instant::now();

        let samples = decrypt_samples(&mut trivium, &msg.trivium_ciphertext);
        let result = model.predict(&samples);

        let result_bytes = vec![bincode::serialize(&result).expect("serialising response")];

        let response = SensorResponse { result_bytes };
        bincode::serialize_into(&mut *stream, &response)?;
        stream.flush()?;

        println!(
            "    Iteration {iteration}: computed in {:.2}s",
            t0.elapsed().as_secs_f64()
        );
    }

    Ok(iteration)
}
