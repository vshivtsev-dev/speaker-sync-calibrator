'use strict';

// The browser's job is narrow: hold the microphone, ship raw samples, and
// render what the server sends back. All measurement happens server-side.

const el = (id) => document.getElementById(id);

// Match the page's own scheme: wss behind the TLS-terminating proxy, ws when
// developing against http://localhost (which browsers count as a secure
// context, so the microphone still works there).
const socketUrl = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`;

let socket = null;
let audio = null;      // AudioContext
let stream = null;     // MediaStream
let node = null;       // AudioWorkletNode
let recording = false;

function setStatus(text, kind) {
  el('status').textContent = text;
  el('dot').className = 'dot' + (kind ? ' ' + kind : '');
}

function setBusy(busy) {
  el('run').disabled = busy;
  el('probe').disabled = busy;
}

function setProgress(fraction) {
  el('bar').style.width = Math.max(0, Math.min(1, fraction)) * 100 + '%';
}

// --------------------------------------------------------------- microphone

async function openMicrophone() {
  if (audio && audio.state !== 'closed') return;

  // Every piece of browser "helpfulness" here is actively harmful. Echo
  // cancellation would suppress precisely the signal being measured, noise
  // suppression would gate the chirp, and automatic gain would move the
  // levels mid-measurement.
  stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
      channelCount: 1,
    },
  });

  audio = new AudioContext();
  await audio.audioWorklet.addModule('/static/recorder-worklet.js');

  node = new AudioWorkletNode(audio, 'spinalign-recorder');
  node.port.onmessage = (event) => {
    if (recording && socket && socket.readyState === WebSocket.OPEN) {
      socket.send(event.data.buffer);
    }
  };

  audio.createMediaStreamSource(stream).connect(node);
  // Keep the graph pulling without routing the microphone to the speakers.
  const sink = audio.createGain();
  sink.gain.value = 0;
  node.connect(sink).connect(audio.destination);
}

// ------------------------------------------------------------------- socket

function connect() {
  socket = new WebSocket(socketUrl);
  socket.binaryType = 'arraybuffer';

  socket.onopen = () => setStatus('Готово. Положите телефон туда, где слушаете.');
  socket.onclose = () => {
    setStatus('Соединение потеряно, переподключаюсь…', 'err');
    setBusy(true);
    setTimeout(connect, 2000);
  };
  socket.onmessage = (event) => handle(JSON.parse(event.data));
}

function handle(message) {
  switch (message.type) {
    case 'record_start':
      recording = true;
      setStatus('Идёт запись…', 'live');
      break;

    case 'record_stop':
      recording = false;
      socket.send(JSON.stringify({
        type: 'recording_done',
        sample_rate: audio ? audio.sampleRate : 48000,
      }));
      setStatus('Считаю…', 'live');
      break;

    case 'progress':
      onProgress(message);
      break;

    case 'sign':
      setStatus(message.detail, message.conclusive ? 'ok' : 'err');
      setBusy(false);
      refreshPlayers();
      break;

    case 'report':
      showReport(message);
      setBusy(false);
      refreshPlayers();
      break;

    case 'error':
      setStatus('Ошибка: ' + message.message, 'err');
      setBusy(false);
      break;
  }
}

let passOffset = 0;
let passSpan = 1;

function onProgress(message) {
  if (message.stage === 'pass') {
    // Two passes when verifying: measure, then prove.
    passOffset = message.which === 'after' ? 0.5 : 0;
    passSpan = message.which === 'after' ? 0.5 : 0.5;
    setStatus(message.which === 'after' ? 'Проверочный замер…' : 'Замер…', 'live');
  } else if (message.stage === 'round') {
    setProgress(passOffset + passSpan * (message.round / message.rounds));
    setStatus(`Играет: ${nameOf(message.player_id)} (${message.round}/${message.rounds})`, 'live');
  } else if (message.stage === 'applying') {
    setStatus(`Применяю поправки (${message.writes})…`, 'live');
  }
}

// ------------------------------------------------------------------ players

let players = [];

function nameOf(playerId) {
  const found = players.find((p) => p.player_id === playerId);
  return found ? found.name : playerId;
}

async function refreshPlayers() {
  const response = await fetch('/api/players');
  const data = await response.json();
  players = data.players;

  const rows = players.map((p) => `
    <tr>
      <td>${escapeHtml(p.name)}</td>
      <td class="num">${p.sync_adjust_ms > 0 ? '+' : ''}${p.sync_adjust_ms} мс</td>
      <td><span class="pill ${p.calibratable ? 'on' : 'off'}">${
        p.calibratable ? 'готова' : (p.provider.startsWith('sendspin') ? 'недоступна' : p.provider)
      }</span></td>
    </tr>`).join('');

  el('players').innerHTML =
    `<thead><tr><th>Колонка</th><th>sync_adjust</th><th></th></tr></thead>
     <tbody>${rows || '<tr><td colspan="3">Sendspin-колонки не найдены</td></tr>'}</tbody>`;

  const ready = players.filter((p) => p.calibratable).length;
  setBusy(ready < 2);
  if (ready < 2) {
    setStatus('Нужно минимум две доступные Sendspin-колонки.', 'err');
  }
}

// ------------------------------------------------------------------- report

function showReport(report) {
  el('result').classList.remove('hidden');
  setProgress(1);

  el('before').textContent = report.spread_before_ms.toFixed(0);
  const after = el('after');
  after.textContent = report.spread_after_ms === null ? '—' : report.spread_after_ms.toFixed(0);
  after.className = 'headline ' + (report.improved ? 'good' : 'bad');

  el('corrections').innerHTML =
    `<thead><tr><th>Колонка</th><th>было</th><th>стало</th><th>ошибка</th></tr></thead><tbody>` +
    report.players.map((p) => `
      <tr>
        <td>${escapeHtml(p.name)}${p.clamped ? ' <span class="pill off">предел</span>' : ''}</td>
        <td class="num">${p.current_adjust_ms} мс</td>
        <td class="num"><b>${p.target_adjust_ms} мс</b></td>
        <td class="num">${p.residual_error_ms.toFixed(1)} мс</td>
      </tr>`).join('') + '</tbody>';

  const notes = [];
  if (report.strategy === 'centered') {
    notes.push(['warn',
      'Разброс больше 500 мс, поэтому поправки отсчитаны от середины, а не от самой медленной колонки.']);
  }
  if (!report.fits) {
    notes.push(['bad',
      'Разброс не влезает в ±500 мс — часть колонок выровнять до конца нельзя.']);
  }
  report.problems.forEach((p) => notes.push(['bad', p]));
  if (!notes.length && report.improved) {
    notes.push(['', 'Проверочный замер подтвердил результат.']);
  }

  el('notes').innerHTML = notes
    .map(([kind, text]) => `<p class="note ${kind}">${escapeHtml(text)}</p>`)
    .join('');

  setStatus(report.improved ? 'Готово.' : 'Завершено с замечаниями.',
            report.improved ? 'ok' : 'err');
}

function escapeHtml(value) {
  const node = document.createElement('span');
  node.textContent = String(value);
  return node.innerHTML;
}

// --------------------------------------------------------------------- boot

async function start(kind) {
  try {
    setBusy(true);
    setProgress(0);
    await openMicrophone();
    if (audio.state === 'suspended') await audio.resume();
    socket.send(JSON.stringify({ type: kind }));
  } catch (error) {
    setStatus('Нет доступа к микрофону: ' + error.message, 'err');
    setBusy(false);
  }
}

el('run').addEventListener('click', () => start('calibrate'));
el('probe').addEventListener('click', () => start('probe_sign'));

refreshPlayers();
connect();
