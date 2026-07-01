#[derive(serde::Serialize, serde::Deserialize)]
pub struct SetupRequest {
    pub server_key_bytes: Vec<u8>,
    /// bincode of `Vec<CompressedFheBool>` (80 items).
    pub encrypted_key_bits: Vec<u8>,
    /// Plaintext IV as 10 bytes (80 bits). Both sides convert to [bool; 80].
    pub iv: [u8; 10],
}

impl SetupRequest {
    pub fn size_of(&self) -> usize {
        self.server_key_bytes.len() + self.encrypted_key_bits.len() + self.iv.len()
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct SetupAck;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct SensorMessage {
    pub trivium_ciphertext: Vec<u16>,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct SensorResponse {
    /// Each entry is the bincode of one `FheUint8`.
    pub result_bytes: Vec<Vec<u8>>,
}
