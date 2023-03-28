import express from 'express';
import fs from 'fs';
import path from 'path';
import * as tf from '@tensorflow/tfjs-node';

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
    console.log(data.shape, data.dtype);
    console.log(data.slice(0, 1000).print());

    const trainLength = Math.floor(0.9 * data.size);
    const trainData = data.slice(0, trainLength);
    const validationData = data.slice(trainLength);

    const contextLength = 8;
    console.log(trainData.slice(0, contextLength + 1).print());
  });
});
