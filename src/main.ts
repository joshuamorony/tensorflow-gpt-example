import express from 'express';
import fs from 'fs';
import path from 'path';
import * as tf from '@tensorflow/tfjs-node';
import {
  Tensor1D,
  Tensor2D,
  Tensor3D,
  TensorLike,
} from '@tensorflow/tfjs-node';
import { log } from 'console';

const host = process.env.HOST ?? 'localhost';
const port = process.env.PORT ? Number(process.env.PORT) : 3333;

const app = express();

// app.get('/', (req, res) => {
// res.send({});
// });

app.listen(port, host, () => {
  console.log(`[ ready ] http://${host}:${port}`);

  const stream = fs.createReadStream(
    path.resolve(__dirname, './assets/tinyshakespeare_input.txt'),
    'utf8'
  );

  const chunks = [];
  stream.on('data', (chunk) => chunks.push(Buffer.from(chunk)));
  stream.on('end', () => {
    // const text = Buffer.concat(chunks).toString();
    // TODO: JUST testing
    const text = 'Wherefore art thou Romeo';

    const set = new Set(text);
    const sorted = Array.from(set).sort((a, b) => (a < b ? -1 : 1));
    const vocabSize = sorted.length;

    // map characters to integers
    const stoi = new Map<string, number>();
    // map integers to characters
    const itos = new Map<number, string>();

    const encode = (val: string) => {
      return Array.from(val).map((char) => stoi.get(char));
    };

    const decode = (val: number[]) => {
      return val.map((encoded) => itos.get(encoded)).join('');
    };

    for (const [index, value] of sorted.entries()) {
      stoi.set(value, index);
      itos.set(index, value);
    }

    const data = tf.tensor1d(encode(text), 'int32');

    const trainLength = Math.floor(0.9 * data.size);
    const trainData = data.slice(0, trainLength);
    const validationData = data.slice(trainLength);

    const blockSize = 8;
    const batchSize = 4;

    const getBatch = (split: string) => {
      const batchData = split === 'train' ? trainData : validationData;

      // Generate random offsets for batch size e.g [2, 3, 5, 6]
      const offsets = tf.randomUniform(
        [batchSize],
        0,
        batchData.size - blockSize,
        'int32',
        1337
      );

      // Use those offsets to get random context and target samples
      const contextBatch: Tensor1D[] = [];
      const targetBatch: Tensor1D[] = [];

      for (const offset of offsets.dataSync()) {
        contextBatch.push(batchData.slice(offset, blockSize));
        targetBatch.push(batchData.slice(offset + 1, blockSize));
      }

      const contexts = tf.stack(contextBatch);
      const targets = tf.stack(targetBatch);

      return { contexts, targets };
    };

    const { contexts, targets } = getBatch('train');
    // console.log('inputs:');
    // console.log(contexts.shape);
    // console.log(contexts.arraySync());
    // console.log('targets:');
    // console.log(targets.shape);
    // console.log(targets.arraySync());
    // console.log('---');

    for (let i = 0; i < batchSize; i++) {
      for (let j = 0; j < blockSize; j++) {
        const context = contexts
          .slice([i, 0], 1)
          .as1D()
          .slice(0, j + 1)
          .dataSync();
        const target = targets.slice([i, 0], 1).as1D().slice(j, 1).dataSync();
        console.log(`When input is ${context} the target is ${target}`);
      }
    }

    const model = new BigramLanguageModel(vocabSize);
    const [logits, loss] = model.forward(contexts, targets);

    // console.log(decode(model.generate(tf.zeros([1, 1]), 100)[0]));
    // const idx = tf.zeros([1, 1], 'int32');
    // const generated = model.generate(idx, 100).arraySync();
    // console.log(decode(generated[0]));
  });
});

class BigramLanguageModel {
  tokenEmbeddingTable: any;
  constructor(private vocabSize: number) {
    this.tokenEmbeddingTable = tf.layers.embedding({
      inputDim: vocabSize,
      outputDim: vocabSize,
    });
  }

  forward(idx, targets = null): [Tensor3D, number] {
    const logits = this.tokenEmbeddingTable.apply(idx);

    let loss = null;

    if (targets !== null) {
      const [T, B, C] = logits.shape;
      const logitsReshaped = logits.reshape([B * T, C]);
      const targetsReshaped = targets.reshape([B * T]);

      const oneHotLabels = tf.oneHot(
        targetsReshaped,
        logitsReshaped.shape[logitsReshaped.rank - 1]
      );

      loss = tf.losses.softmaxCrossEntropy(oneHotLabels, logitsReshaped);

      // const oneHotLabels = tf.oneHot(targets, logits.shape[logits.rank - 1]);
      // loss = tf.losses.softmaxCrossEntropy(oneHotLabels, logits);
    }

    return [logits, loss];
  }

  generate(idx, maxNewTokens) {
    // for (let i = 0; i < maxNewTokens; i++) {
    // // Get the predictions
    // const [logits, loss] = this.forward(idx);
    // console.log('logits');
    // console.log(logits.print());
    // // Split the logits tensor along the time axis
    // const timeAxis = 1;
    // const logitsSlices = tf.split(logits, logits.shape[timeAxis], timeAxis);
    // console.log('slices');
    // console.log(logitsSlices);
    // // Focus only on the last time step
    // const logitsLastStep = logitsSlices[logitsSlices.length - 1].reshape([
    // logits.shape[0],
    // logits.shape[2],
    // ]);
    // // Apply softmax to get probabilities
    // const probs = logitsLastStep.softmax(-1);
    // console.log('probs');
    // console.log(probs.print());
    // // Sample from the distribution
    // // const idxNext = tf.multinomial(probs, 1);
    // // console.log('next');
    // // console.log(idxNext);
    // // // Append sampled index to the running sequence
    // // idx = tf.concat([idx, idxNext], 1);
    // // console.log('new');
    // // console.log(idx);
    // }
    // return idx;
    // console.log('IDX');
    // console.log(idx.print());
    // for (let i = 0; i < maxNewTokens; i++) {
    // // get predictions
    // let [logits, loss] = this.forward(idx);
    // console.log('preslice');
    // console.log(logits.print());
    // console.log(loss);
    // // focus on last time step
    // // TODO: check this - think this is incorrect for my setup
    // logits = logits.slice([0, logits.shape[1] - 1, 0], [-1, 1, -1]); // becomes (B, C)
    // console.log('sliced');
    // console.log(logits.print());
    // // apply softmax to get probabilities
    // const probs = tf.softmax(logits, -1);
    // console.log('probs');
    // console.log(probs.print());
    // // sample from distribution
    // const idx_next = tf.multinomial(probs, 1);
    // console.log('multinomial');
    // console.log(idx_next.print());
    // idx = tf.concat([idx, idx_next], 1); // (B, T+1)
    // console.log('new');
    // console.log(idx);
    // }
    // return idx;
  }
}
