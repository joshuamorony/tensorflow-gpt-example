import express from 'express';
import fs from 'fs';
import path from 'path';
import * as tf from '@tensorflow/tfjs-node';
import { Tensor1D } from '@tensorflow/tfjs-node';

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

    console.log(sorted);
    console.log(vocabSize);

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
    // console.log('TENSOR1D');
    // console.log(data.dataSync());
    // console.log(data.shape, data.dtype);
    // console.log('end print');
    // console.log(data.slice(0, 1000).print());

    const trainLength = Math.floor(0.9 * data.size);
    const trainData = data.slice(0, trainLength);
    const validationData = data.slice(trainLength);

    const blockSize = 8;
    const batchSize = 4;

    // const testX = trainData.slice(0, blockSize);
    // const testY = trainData.slice(1, blockSize + 1);

    // for (let i = 0; i < blockSize; i++) {
    // const context = testX.slice(0, i + 1).dataSync();
    // const target = testY.slice(i, 1).dataSync();
    // console.log(`When input is ${context} the target is ${target}`);
    // }

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
    console.log('inputs:');
    console.log(contexts.shape);
    console.log(contexts.arraySync());
    console.log('targets:');
    console.log(targets.shape);
    console.log(targets.arraySync());
    console.log('---');

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
  });
});
