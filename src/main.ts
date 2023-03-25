import express from 'express';
import fs from 'fs';
import path from 'path';

const host = process.env.HOST ?? 'localhost';
const port = process.env.PORT ? Number(process.env.PORT) : 3333;

const app = express();
let data = '';

app.get('/', (req, res) => {
  res.send({ message: data });
});

app.listen(port, host, () => {
  console.log(`[ ready ] http://${host}:${port}`);

  const stream = fs.createReadStream(
    path.resolve(__dirname, './assets/tinyshakespeare_input.txt'),
    'utf8'
  );

  const chunks = [];
  stream.on('data', (chunk) => chunks.push(Buffer.from(chunk)));
  stream.on('end', () => {
    data = Buffer.concat(chunks).toString();

    const set = new Set(data);
    const sorted = Array.from(set).sort((a, b) => (a < b ? -1 : 1));
    const vocabSize = sorted.length;

    console.log(sorted);
    console.log(vocabSize);
  });
});
