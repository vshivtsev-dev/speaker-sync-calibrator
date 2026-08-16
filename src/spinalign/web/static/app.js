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

let busy = false;      // a measurement is running
let runnable = false;  // enough speakers are switched on to run one

// Kept apart deliberately. Switching a speaker off can leave too few to
// calibrate, and if that disabled the switches too, the user would have no way
// to switch it back on again.
function refreshControls() {
  el('run').disabled = busy || !runnable;
  el('probe').disabled = busy || !runnable;
  document.querySelectorAll('#players .toggle').forEach((b) => { b.disabled = busy; });
}

function setBusy(value) {
  busy = value;
  refreshControls();
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

  // The player list owns the status line: it is the thing that knows whether
  // a calibration can actually start. Announcing "ready" here would overwrite
  // an explanation of why the buttons are disabled.
  socket.onopen = () => refreshPlayers();
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
      // A result now exists, so saving it as a position becomes possible.
      refreshProfiles();
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

// ----------------------------------------------------------- round length

// Filled from the server so the page never invents session parameters, and so
// a value chosen on one device shows up on the next.
let session = { chirps_per_round: 5, guard_chirps: 2, period_seconds: 1.3 };

function chosenChirps() {
  const asked = parseInt(el('chirps').value, 10);
  return Number.isFinite(asked) ? asked : session.chirps_per_round;
}

function describeRoundLength() {
  const chirps = chosenChirps();
  const readings = chirps - session.guard_chirps;
  const ready = players.filter((p) => p.calibratable).length;

  if (readings < 1) {
    el('chirps-note').textContent =
      `Первые ${session.guard_chirps} свиста в каждом круге отбрасываются — их заглушает`
      + ' переключение. Нужно больше.';
    return;
  }

  // One round per speaker, plus a repeat of the first to measure clock drift,
  // and the whole thing runs twice: measure, then verify.
  const rounds = ready ? ready + 1 : 0;
  const seconds = Math.round(rounds * chirps * session.period_seconds * 2);
  const duration = rounds ? `, замер с проверкой ≈ ${seconds} с` : '';
  el('chirps-note').textContent =
    `${readings} ${plural(readings, 'отсчёт', 'отсчёта', 'отсчётов')} на колонку${duration}.`
    + ' Больше свистов — устойчивее к шуму и дольше.';
}

function plural(count, one, few, many) {
  const mod100 = count % 100;
  if (mod100 >= 11 && mod100 <= 14) return many;
  const mod10 = count % 10;
  if (mod10 === 1) return one;
  if (mod10 >= 2 && mod10 <= 4) return few;
  return many;
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
  session = data;

  const chirps = el('chirps');
  chirps.min = data.chirps_range[0];
  chirps.max = data.chirps_range[1];
  // Only while the field is not being edited, or typing would fight the poll.
  if (document.activeElement !== chirps) chirps.value = data.chirps_per_round;

  const rows = players.map((p) => `
    <tr class="${p.enabled ? '' : 'off'}">
      <td>${escapeHtml(p.name)}<br><span class="sub">${escapeHtml(p.transport)}</span>${
        // When the delay setting was not found, show what the server did
        // report: seeing the real names turns a mystery into a one-line fix.
        !p.calibratable && p.config_keys && p.config_keys.length
          ? `<br><span class="sub">настройки: ${escapeHtml(p.config_keys.join(', '))}</span>`
          : ''
      }</td>
      <td class="num">${p.calibratable
        ? `${p.sync_adjust_ms > 0 ? '+' : ''}${p.sync_adjust_ms} мс`
        : '—'}</td>
      <td>
        <span class="pill ${p.calibratable ? 'on' : 'off'}">${
          escapeHtml(p.calibratable ? 'готова' : (p.excluded_because || 'не участвует'))
        }</span>
        <button class="toggle" data-toggle="${escapeAttr(p.player_id)}"
                data-enable="${p.enabled ? '0' : '1'}"${busy ? ' disabled' : ''}>${
          p.enabled ? 'выключить' : 'включить'
        }</button>
      </td>
    </tr>`).join('');

  el('players').innerHTML =
    `<thead><tr><th>Колонка</th><th>sync_adjust</th><th></th></tr></thead>
     <tbody>${rows || '<tr><td colspan="3">Music Assistant не отдал ни одного плеера</td></tr>'}</tbody>`;

  const ready = players.filter((p) => p.calibratable);
  runnable = ready.length >= 2;
  refreshControls();
  describeRoundLength();

  if (!runnable) {
    setStatus(
      players.length
        ? 'Нужно минимум две включённые колонки. Причины — в списке выше.'
        : 'Music Assistant не отдал ни одного плеера.',
      'err',
    );
  } else if (ready.some((p) => !p.is_sendspin)) {
    // Sendspin guarantees the tightest playback sync; other providers still
    // work, since any offset they add is measured and corrected, but the
    // result is only as steady as their own synchronisation.
    setStatus(
      `Готово, колонок: ${ready.length}. Часть из них не Sendspin — точность будет зависеть от их собственной синхронизации.`,
    );
  } else {
    setStatus(`Готово, колонок: ${ready.length}. Положите телефон туда, где слушаете.`);
  }
}

async function setPlayerEnabled(playerId, enabled) {
  try {
    const response = await fetch(`/api/players/${encodeURIComponent(playerId)}/enabled`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled }),
    });
    if (!response.ok) throw new Error((await response.text()).trim());
  } catch (error) {
    setStatus('Не переключилось: ' + error.message, 'err');
    return;
  }
  // The list owns the status line, so it has the last word on what the
  // switch changed — including whether a run is still possible.
  await refreshPlayers();
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

// ----------------------------------------------------------------- profiles

async function refreshProfiles() {
  const response = await fetch('/api/profiles');
  if (!response.ok) {
    // No state directory configured: the feature is simply off.
    el('profiles-card').classList.add('hidden');
    return;
  }

  const data = await response.json();
  el('profiles-card').classList.remove('hidden');
  el('save').disabled = !data.can_save;

  if (!data.profiles.length) {
    el('profiles').innerHTML =
      '<tbody><tr><td>Пока ничего не сохранено</td></tr></tbody>';
    return;
  }

  el('profiles').innerHTML = '<tbody>' + data.profiles.map((p) => `
    <tr>
      <td>${escapeHtml(p.name)}<br><span class="pill">${p.speakers} колонки</span></td>
      <td class="actions">
        <button data-apply="${escapeAttr(p.name)}">Применить</button>
        <button class="link" data-delete="${escapeAttr(p.name)}">Удалить</button>
      </td>
    </tr>`).join('') + '</tbody>';
}

async function applyProfile(name) {
  setBusy(true);
  setStatus(`Применяю «${name}»…`, 'live');
  try {
    const response = await fetch(`/api/profiles/${encodeURIComponent(name)}/apply`, {
      method: 'POST',
    });
    if (!response.ok) throw new Error(await response.text());

    const result = await response.json();
    const written = result.applied.length;
    setStatus(
      result.problems.length
        ? `Применено (${written}), но: ${result.problems.join('; ')}`
        : `Позиция «${name}» применена, изменено колонок: ${written}.`,
      result.problems.length ? 'err' : 'ok',
    );
  } catch (error) {
    setStatus('Не удалось применить: ' + error.message, 'err');
  } finally {
    setBusy(false);
    refreshPlayers();
  }
}

async function saveProfile() {
  const name = el('save-name').value.trim();
  if (!name) return;

  const response = await fetch('/api/profiles', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  });
  if (response.ok) {
    el('save-name').value = '';
    setStatus(`Позиция «${name}» сохранена.`, 'ok');
    refreshProfiles();
  } else {
    setStatus('Не сохранилось: ' + (await response.text()), 'err');
  }
}

async function deleteProfile(name) {
  await fetch(`/api/profiles/${encodeURIComponent(name)}`, { method: 'DELETE' });
  refreshProfiles();
}

function escapeAttr(value) {
  return String(value).replace(/&/g, '&amp;').replace(/"/g, '&quot;');
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
    socket.send(JSON.stringify({ type: kind, chirps_per_round: chosenChirps() }));
  } catch (error) {
    setStatus('Нет доступа к микрофону: ' + error.message, 'err');
    setBusy(false);
  }
}

el('chirps').addEventListener('input', describeRoundLength);
el('run').addEventListener('click', () => start('calibrate'));
el('probe').addEventListener('click', () => start('probe_sign'));
el('save').addEventListener('click', saveProfile);

// Delegated so the list can be re-rendered without rebinding every row.
el('players').addEventListener('click', (event) => {
  const toggle = event.target.closest('[data-toggle]');
  if (toggle) setPlayerEnabled(toggle.dataset.toggle, toggle.dataset.enable === '1');
});

el('profiles').addEventListener('click', (event) => {
  const apply = event.target.closest('[data-apply]');
  if (apply) return applyProfile(apply.dataset.apply);

  const remove = event.target.closest('[data-delete]');
  if (remove) return deleteProfile(remove.dataset.delete);
});

refreshPlayers();
refreshProfiles();
connect();
