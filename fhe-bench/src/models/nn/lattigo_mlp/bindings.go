package main

/*
#include <stdlib.h>
*/
import "C"

import (
	"sync"
	"unsafe"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"lattigo_mlp/mlpmodel"
)

type mlpContext struct {
	bsEv      *bootstrapping.Evaluator
	ev        *ckks.Evaluator
	param     ckks.Parameters
	encoder   *ckks.Encoder
	encryptor *rlwe.Encryptor
	decryptor *rlwe.Decryptor
}

var (
	handleMu sync.Mutex
	handles  = map[int64]any{}
	handleN  int64
)

func putHandle(v any) int64 {
	handleMu.Lock()
	defer handleMu.Unlock()
	handleN++
	handles[handleN] = v
	return handleN
}

func getHandle(id int64) any {
	handleMu.Lock()
	defer handleMu.Unlock()
	return handles[id]
}

//export MlpConfigure
func MlpConfigure() C.longlong {
	bsEv, ev, param, encoder, encryptor, decryptor := mlpmodel.Configure()
	ctx := &mlpContext{bsEv, ev, param, encoder, encryptor, decryptor}
	return C.longlong(putHandle(ctx))
}

//export MlpEncryptInput
func MlpEncryptInput(ctxID C.longlong, data *C.float, n C.int) C.longlong {
	ctx := getHandle(int64(ctxID)).(*mlpContext)
	src := unsafe.Slice((*float32)(unsafe.Pointer(data)), int(n))
	input := make([]float32, len(src))
	copy(input, src)
	cts := mlpmodel.EncryptInput(ctx.ev, ctx.param, ctx.encoder, ctx.encryptor, input)
	return C.longlong(putHandle(cts))
}

//export MlpEvaluate
func MlpEvaluate(ctxID C.longlong, ctsID C.longlong) C.longlong {
	ctx := getHandle(int64(ctxID)).(*mlpContext)
	cts := getHandle(int64(ctsID)).([]*rlwe.Ciphertext)
	result := mlpmodel.Evaluate(ctx.bsEv, ctx.ev, ctx.param, ctx.encoder, cts)
	return C.longlong(putHandle(result))
}

//export MlpDecryptResult
func MlpDecryptResult(ctxID C.longlong, ctsID C.longlong, out *C.float, outLen C.int) {
	ctx := getHandle(int64(ctxID)).(*mlpContext)
	cts := getHandle(int64(ctsID)).([]*rlwe.Ciphertext)
	vals := mlpmodel.DecryptResult(ctx.ev, ctx.param, ctx.encoder, ctx.decryptor, cts)
	dst := unsafe.Slice((*float32)(unsafe.Pointer(out)), int(outLen))
	copy(dst, vals[:min(len(vals), int(outLen))])
}

//export MlpFree
func MlpFree(id C.longlong) {
	handleMu.Lock()
	defer handleMu.Unlock()
	delete(handles, int64(id))
}
