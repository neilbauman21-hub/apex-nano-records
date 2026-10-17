import puppeteer from 'puppeteer-core';
import http from 'http';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const WEB_DIR = path.join(__dirname, 'web');
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

const server = http.createServer((req, res) => {
  const cleanUrl = req.url === '/' ? '/index.html' : req.url.split('?')[0];
  const filePath = path.join(WEB_DIR, cleanUrl);
  if (!fs.existsSync(filePath)) {
    res.writeHead(404);
    res.end('Not found');
    return;
  }
  const ext = path.extname(filePath);
  const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json' };
  res.writeHead(200, { 'Content-Type': types[ext] || 'text/plain' });
  fs.createReadStream(filePath).pipe(res);
});

server.listen(0, '127.0.0.1', async () => {
  const { port } = server.address();
  const url = `http://127.0.0.1:${port}/`;
  console.log(`Serving APEX // NANO-RECORDS dashboard at ${url}`);

  const browser = await puppeteer.launch({
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--window-size=1520,1080'],
    defaultViewport: { width: 1520, height: 1080, deviceScaleFactor: 2 },
  });

  const page = await browser.newPage();
  const errors = [];
  page.on('console', msg => { if (msg.type() === 'error') errors.push(msg.text()); });
  page.on('pageerror', err => errors.push(err.message));

  await page.goto(url, { waitUntil: 'networkidle0' });
  await sleep(600);

  // Screenshot 1: Main Swiss Monograph Dashboard (4p Active)
  await page.screenshot({ path: path.join(__dirname, 'screenshot-1-monograph-4p.png'), fullPage: true });
  console.log('Saved screenshot-1-monograph-4p.png');

  // Switch to 4p Multiplier tab & trigger max 9^20 stress test
  await page.click('.model-tab[data-model="mult4p"]');
  await page.click('#btn-stress-carry');
  await sleep(250);

  // Exercise Omega-rANS compressor and ARC-ZERO interactive task selector
  await page.click('#btn-run-compress');
  await sleep(150);
  await page.click('#arc-task-tabs .model-tab[data-arc="1"]');
  await sleep(150);

  // Switch to 28p Trained tab & trigger stress carry test
  await page.click('.model-tab[data-model="28p"]');
  await page.click('#btn-stress-carry');
  await sleep(300);
  await page.screenshot({ path: path.join(__dirname, 'screenshot-2-trained-28p.png'), fullPage: true });
  console.log('Saved screenshot-2-trained-28p.png');

  // Open Mathematical Proof Dialog
  await page.click('#btn-open-proof');
  await sleep(300);
  await page.screenshot({ path: path.join(__dirname, 'screenshot-3-proofs-dialog.png') });
  console.log('Saved screenshot-3-proofs-dialog.png');

  await browser.close();
  server.close();

  if (errors.length > 0) {
    console.error('Console/Page errors:', errors);
    process.exit(1);
  }
  console.log('All APEX // NANO-RECORDS dashboard checks passed with ZERO errors!');
});
