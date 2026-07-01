package mlpmodel

import (
	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func Configure() (*bootstrapping.Evaluator, *ckks.Evaluator, ckks.Parameters, *ckks.Encoder, *rlwe.Encryptor, *rlwe.Decryptor) {
	return mlp__configure()
}

func EncryptInput(ev *ckks.Evaluator, param ckks.Parameters, encoder *ckks.Encoder, encryptor *rlwe.Encryptor, input []float32) []*rlwe.Ciphertext {
	return mlp__encrypt__arg0(ev, param, encoder, encryptor, input)
}

func Evaluate(bsEv *bootstrapping.Evaluator, ev *ckks.Evaluator, param ckks.Parameters, encoder *ckks.Encoder, cts []*rlwe.Ciphertext) []*rlwe.Ciphertext {
	return mlp(bsEv, ev, param, encoder, cts)
}

func DecryptResult(ev *ckks.Evaluator, param ckks.Parameters, encoder *ckks.Encoder, decryptor *rlwe.Decryptor, cts []*rlwe.Ciphertext) []float32 {
	return mlp__decrypt__result0(ev, param, encoder, decryptor, cts)
}
