// Fixed response server only. No DNS, forwarding, CONNECT, upstream fetch or fallback.
const fs = require('fs');
const path = require('path');
const https = require('https');
const crypto = require('crypto');
const root = process.argv[2];
const port = Number(process.argv[3] || 443);
const ready = process.argv[4];
const manifest = JSON.parse(fs.readFileSync(path.join(root, 'manifest.json')));
const responses = new Map();
for (const [url, item] of Object.entries(manifest.resources)) {
  const bytes = fs.readFileSync(path.join(root, 'blobs', item.sha256 + '.bin'));
  if (bytes.length !== item.size || crypto.createHash('sha256').update(bytes).digest('hex') !== item.sha256)
    throw Error('offline response hash mismatch');
  responses.set(url, { bytes, contentType: item.content_type });
}
const server = https.createServer({
  key: fs.readFileSync(path.join(root, 'server.key')),
  cert: fs.readFileSync(path.join(root, 'server.pem')),
  minVersion: 'TLSv1.2', maxHeaderSize: 8192,
}, (req, res) => {
  req.on('error', () => res.destroy());
  const host = String(req.headers.host || '').toLowerCase().replace(/:443$/, '');
  const item = responses.get('https://' + host + req.url);
  if (!['GET', 'HEAD'].includes(req.method)) { res.writeHead(405); res.end(); return; }
  if (!item) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, {'Content-Type': item.contentType, 'Content-Length': item.bytes.length,
    'Access-Control-Allow-Origin': '*', 'Cache-Control': 'no-store'});
  res.end(req.method === 'HEAD' ? undefined : item.bytes);
});
server.headersTimeout = 5000;
server.requestTimeout = 10000;
server.on('clientError', (_, socket) => socket.destroy());
server.listen(port, '127.0.0.1', () => {
  fs.writeFileSync(ready, String(server.address().port));
  console.log('MMPTB_OFFLINE_READY resources=' + responses.size);
});
