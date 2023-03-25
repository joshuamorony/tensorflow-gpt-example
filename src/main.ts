import express from 'express';
import fs from 'fs';
import path from 'path';

const host = process.env.HOST ?? 'localhost';
const port = process.env.PORT ? Number(process.env.PORT) : 3333;

const app = express();

app.get('/', (req, res) => {
  res.send({ message: 'Hello API' });
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
    const data = Buffer.concat(chunks).toString();
    console.log(data);
  });
});
