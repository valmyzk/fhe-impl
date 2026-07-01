package main

import (
	"encoding/json"
	"fmt"
	"os"
	"time"

	"lattigo_mlp/mlpmodel"
)

func logf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
}

func main() {
	logf("Setting up HE parameters (this may take several minutes)...")
	start := time.Now()
	bsEv, ev, param, encoder, encryptor, decryptor := mlpmodel.Configure()
	logf("Setup complete in %s", time.Since(start).Round(time.Second))

	var input []float32
	if err := json.NewDecoder(os.Stdin).Decode(&input); err != nil {
		fmt.Fprintf(os.Stderr, "error reading input: %v\n", err)
		os.Exit(1)
	}
	logf("Input: %v", input)

	logf("Encrypting input...")
	cts := mlpmodel.EncryptInput(ev, param, encoder, encryptor, input)

	logf("Running homomorphic inference...")
	start = time.Now()
	resultCts := mlpmodel.Evaluate(bsEv, ev, param, encoder, cts)
	logf("Inference complete in %s", time.Since(start).Round(time.Second))

	logf("Decrypting result...")
	output := mlpmodel.DecryptResult(ev, param, encoder, decryptor, resultCts)

	fmt.Printf("%f\n", output[0])
}
