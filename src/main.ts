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
    const text = Buffer.concat(chunks).toString();
    // TODO: JUST testing
    // const text = 'Wherefore art thou Romeo';

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
    let batchSize = 32;

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

    // original junk generation
    // zeros here is (B, T)
    // basically we will have [[0]] to kick off generation, 0 is space character - good starting point
    // const idx = tf.zeros([1, 1], 'int32');
    // console.log('call generate');
    // const generated = model.generate(idx, 100).arraySync();
    // console.log(decode(generated[0]));

    const optimiser = tf.train.adam(1e-3);

    // training loop
    for (let i = 0; i < 1000; i++) {
      const { contexts, targets } = getBatch('train');

      optimiser.minimize(() => {
        const [logits, loss] = model.forward(contexts, targets);

        // show progression after each training loop
        console.log(loss.arraySync());
        const idx = tf.zeros([1, 1], 'int32');
        const generated = model.generate(idx, 200).arraySync();
        console.log(decode(generated[0]));

        return loss;
      });
    }

    // optimiser.getWeights().then((weights) => {
    // const trainableWeights = model.tokenEmbeddingTable.trainableWeights;
    // const weightTensors = [
    // weights.slice(0, trainableWeights[0].shape.length),
    // ];
    // model.tokenEmbeddingTable.setWeights(weightTensors);

    // // after training generation
    // const idx = tf.zeros([1, 1], 'int32');
    // console.log("training complete... let's generate some text!");
    // const generated = model.generate(idx, 200).arraySync();
    // console.log(decode(generated[0]));
    // });
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

  forward(idx, targets = null) {
    // Both idx and targets are (B, T) tensors
    const logits = this.tokenEmbeddingTable.apply(idx); // (B, T, C) e.g. (4, 8, 12)

    let loss = null;

    if (targets !== null) {
      const oneHotLabels = tf.oneHot(targets, logits.shape[logits.rank - 1]);
      loss = tf.losses.softmaxCrossEntropy(oneHotLabels, logits);
    }

    return [logits, loss];
  }

  generate(idx, maxNewTokens) {
    for (let i = 0; i < maxNewTokens; i++) {
      // get predictions
      let [logits, loss] = this.forward(idx);

      const lastTimeStep = logits
        .slice([0, logits.shape[1] - 1, 0], [-1, 1, -1])
        .reshape([-1, logits.shape[2]]);

      const probs = tf.softmax(lastTimeStep, -1);
      const idx_next = tf.multinomial(probs, 1);
      idx = tf.concat([idx, idx_next], 1); // (B, T+1)
    }
    return idx;
  }
}
