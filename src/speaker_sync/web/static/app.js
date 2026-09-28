'use strict';

// The browser's job is narrow: hold the microphone, ship raw samples, and
// render what the server sends back. All measurement happens server-side.

const el = (id) => document.getElementById(id);

// ------------------------------------------------------------------ language

// The server resolves the language — the configured setting, or the browser's
// own preference under "auto" — and states it on the page, so the text drawn
// here and the messages the server sends always agree.
const LANG = document.documentElement.lang === 'ru' ? 'ru' : 'en';

// Russian needs three forms after a number; English two.
function plural(count, forms) {
  if (LANG !== 'ru') return count === 1 ? forms[0] : forms[1];
  const mod100 = count % 100;
  if (mod100 >= 11 && mod100 <= 14) return forms[2];
  const mod10 = count % 10;
  if (mod10 === 1) return forms[0];
  if (mod10 >= 2 && mod10 <= 4) return forms[1];
  return forms[2];
}

const STRINGS = {
  en: {
    tagline: 'Align your speakers\' delays using your phone\'s microphone',
    popout_text: 'The microphone may be unavailable inside the Home Assistant panel —',
    popout_link: 'open Speaker Sync Calibrator in a separate tab',
    speakers: 'Speakers',
    refresh: 'Refresh',
    loading: 'Loading…',
    switch_note: 'A speaker switched off is left alone: it is not measured, not grouped'
      + ' and its delay is not changed. Useful for a subwoofer, a speaker in another'
      + ' room, or one you set by hand.',
    idle: 'Put the phone where you listen and press “Calibrate”.',
    chirps_label: 'Chirps per speaker',
    run: 'Calibrate',
    run_note: 'It will be quiet at first, then the speakers take turns playing short'
      + ' chirps. Keep the phone still and the room quiet. Whatever was playing stops:'
      + ' the track is queued as ordinary playback — only that way do mute and grouping,'
      + ' which the measurement relies on, work.',
    result: 'Result',
    ms_spread: 'ms spread',
    check: 'Check without changes',
    check_note: '“Check” measures the speakers as they are now, at 44.1 and 48 kHz, with'
      + ' every speaker together at the end — and changes nothing. It shows whether the'
      + ' corrections hold, and whether a speaker\'s delay depends on the track format.',
    listen: 'Listen to clicks',
    listen_note: '“Listen” plays clicks on every speaker at once for about 20 s — no'
      + ' microphone. Stand where the phone was. One sharp click: in sync. A thick or'
      + ' ringing click: a few ms apart. A double “tr-ack”: more than ~10 ms, calibrate.',
    listening: 'Clicks are playing on every speaker. One sharp click means in sync;'
      + ' a double click means they are apart.',
    check_result: 'Check',
    offset_from_first: 'after the earliest',
    in_sync: 'In sync: the corrections hold, in both formats and with every speaker playing.',
    save_placeholder: 'Position name, e.g. “sofa”',
    save: 'Save',
    positions: 'Positions',
    positions_note: 'A calibration belongs to the spot the phone was in: it also'
      + ' compensates for sound travelling through the air. The sofa and the kitchen need'
      + ' different corrections — a saved position is applied without a new measurement.',
    disconnected: 'Connection lost, reconnecting…',
    recording: 'Recording…',
    computing: 'Computing…',
    error: 'Error: ',
    verifying: 'Verification pass…',
    measuring: 'Measuring…',
    playing: (v) => `Playing: ${v.name} (${v.round}/${v.rounds})`,
    everyone: 'every speaker at once',
    applying: (v) => `Applying corrections (${v.writes})…`,
    guard_too_many: (v) => `The first ${v.guard} chirps of each round are discarded — the`
      + ' switch-over masks them. More are needed.',
    readings: (v) => `${v.n} ${plural(v.n, ['reading', 'readings'])} per speaker.`,
    total: (v) => ` ${v.total} ${plural(v.total, ['chirp', 'chirps'])} in all:`
      + ' the first speaker plays twice, and the repeat measures the phone clock\'s drift.'
      + ` Measurement with verification ≈ ${v.seconds} s.`,
    settings_found: 'delay setting not found — show what the player has',
    settings_all: 'player settings',
    ms: 'ms',
    ready_pill: 'ready',
    not_taking_part: 'not taking part',
    switch_off: 'switch off',
    switch_on: 'switch on',
    speaker: 'Speaker',
    no_players: 'Music Assistant reported no players',
    need_two: 'At least two speakers need to be switched on. The reasons are in the list above.',
    no_players_status: 'Music Assistant reported no players.',
    ready_mixed: (v) => `Ready, ${v.n} ${plural(v.n, ['speaker', 'speakers'])}. Some of them`
      + ' are not Sendspin — accuracy will depend on their own synchronisation.',
    ready: (v) => `Ready, ${v.n} ${plural(v.n, ['speaker', 'speakers'])}.`
      + ' Put the phone where you listen.',
    toggle_failed: 'Did not switch: ',
    was: 'was',
    now: 'now',
    residual: 'error',
    limit: 'limit',
    centered: 'The spread cannot be covered by a one-sided delay, so the corrections are'
      + ' measured from the middle rather than from the slowest speaker.',
    fastest: 'The delay setting on these speakers can only advance, not delay, so'
      + ' everyone is pulled up to the fastest speaker.',
    does_not_fit: 'The spread does not fit within ±500 ms — some speakers cannot be'
      + ' fully aligned.',
    verified: 'The verification pass confirmed the result.',
    together_ok: (v) => `With every speaker playing at once: ${v.spread} ms apart, as measured one by one.`,
    done: 'Done.',
    done_with_notes: 'Finished with remarks.',
    nothing_saved: 'Nothing saved yet',
    n_speakers: (v) => `${v.n} ${plural(v.n, ['speaker', 'speakers'])}`,
    apply: 'Apply',
    delete: 'Delete',
    applying_position: (v) => `Applying “${v.name}”…`,
    applied_with_problems: (v) => `Applied (${v.n}), but: ${v.problems}`,
    applied: (v) => `Position “${v.name}” applied, speakers changed: ${v.n}.`,
    apply_failed: 'Could not apply: ',
    saved: (v) => `Position “${v.name}” saved.`,
    save_failed: 'Not saved: ',
    no_microphone: 'No access to the microphone: ',
  },
  ru: {
    tagline: 'Выравнивание задержек колонок по микрофону телефона',
    popout_text: 'Микрофон внутри панели Home Assistant может быть недоступен —',
    popout_link: 'откройте Speaker Sync Calibrator в отдельной вкладке',
    speakers: 'Колонки',
    refresh: 'Обновить',
    loading: 'Загрузка…',
    switch_note: 'Выключенную колонку калибровка не трогает: её не замеряют, в группу не'
      + ' берут и задержку ей не меняют. Пригодится для сабвуфера, колонки в другой'
      + ' комнате или той, что настроена вручную.',
    idle: 'Положите телефон туда, где слушаете, и нажмите «Калибровать».',
    chirps_label: 'Свистов на колонку',
    run: 'Калибровать',
    run_note: 'Во время замера будет тихо, потом колонки по очереди издадут короткие'
      + ' свисты. Не двигайте телефон и старайтесь не шуметь. То, что играло,'
      + ' остановится: трек ставится в очередь как обычное воспроизведение — только так'
      + ' работают мьют и группа, на которых держится замер.',
    result: 'Результат',
    ms_spread: 'мс разброса',
    check: 'Проверить без изменений',
    check_note: '«Проверить» меряет колонки как есть, на 44,1 и 48 кГц, в конце — все вместе,'
      + ' и ничего не меняет. Видно, держатся ли поправки и не зависит ли задержка'
      + ' колонки от формата трека.',
    listen: 'Послушать щелчки',
    listen_note: '«Послушать» играет щелчки на всех колонках сразу, около 20 с, без'
      + ' микрофона. Встаньте туда, где лежал телефон. Один чёткий щелчок — синхронно.'
      + ' «Толстый» или звенящий — расхождение в несколько мс. Двойной «тр-ак» — больше'
      + ' ~10 мс, нужна калибровка.',
    listening: 'Щелчки играют на всех колонках. Один чёткий щелчок — синхронно;'
      + ' двойной — колонки расходятся.',
    check_result: 'Проверка',
    offset_from_first: 'после самой ранней',
    in_sync: 'Синхронно: поправки держатся — в обоих форматах и когда играют все сразу.',
    save_placeholder: 'Название позиции, например «диван»',
    save: 'Сохранить',
    positions: 'Позиции',
    positions_note: 'Калибровка привязана к точке, где стоял телефон: компенсируется в том'
      + ' числе путь звука по воздуху. Для дивана и кухни нужны разные поправки —'
      + ' сохранённую позицию можно применить без нового замера.',
    disconnected: 'Соединение потеряно, переподключаюсь…',
    recording: 'Идёт запись…',
    computing: 'Считаю…',
    error: 'Ошибка: ',
    verifying: 'Проверочный замер…',
    measuring: 'Замер…',
    playing: (v) => `Играет: ${v.name} (${v.round}/${v.rounds})`,
    everyone: 'все колонки сразу',
    applying: (v) => `Применяю поправки (${v.writes})…`,
    guard_too_many: (v) => `Первые ${v.guard} свиста в каждом круге отбрасываются — их`
      + ' заглушает переключение. Нужно больше.',
    readings: (v) => `${v.n} ${plural(v.n, ['отсчёт', 'отсчёта', 'отсчётов'])} на колонку.`,
    total: (v) => ` Всего ${v.total} ${plural(v.total, ['свист', 'свиста', 'свистов'])}:`
      + ' первая колонка звучит дважды, по повтору измеряется уход часов телефона.'
      + ` Замер с проверкой ≈ ${v.seconds} с.`,
    settings_found: 'настройка задержки не найдена — показать настройки плеера',
    settings_all: 'настройки плеера',
    ms: 'мс',
    ready_pill: 'готова',
    not_taking_part: 'не участвует',
    switch_off: 'выключить',
    switch_on: 'включить',
    speaker: 'Колонка',
    no_players: 'Music Assistant не отдал ни одного плеера',
    need_two: 'Нужно минимум две включённые колонки. Причины — в списке выше.',
    no_players_status: 'Music Assistant не отдал ни одного плеера.',
    ready_mixed: (v) => `Готово, колонок: ${v.n}. Часть из них не Sendspin — точность`
      + ' будет зависеть от их собственной синхронизации.',
    ready: (v) => `Готово, колонок: ${v.n}. Положите телефон туда, где слушаете.`,
    toggle_failed: 'Не переключилось: ',
    was: 'было',
    now: 'стало',
    residual: 'ошибка',
    limit: 'предел',
    centered: 'Разброс не покрывается односторонней задержкой, поэтому поправки отсчитаны'
      + ' от середины, а не от самой медленной колонки.',
    fastest: 'Настройка задержки на этих колонках умеет только торопить, но не задерживать,'
      + ' поэтому все подтянуты к самой быстрой колонке.',
    does_not_fit: 'Разброс не влезает в ±500 мс — часть колонок выровнять до конца нельзя.',
    verified: 'Проверочный замер подтвердил результат.',
    together_ok: (v) => `Когда играют все сразу: расхождение ${v.spread} мс, как и по одной.`,
    done: 'Готово.',
    done_with_notes: 'Завершено с замечаниями.',
    nothing_saved: 'Пока ничего не сохранено',
    n_speakers: (v) => `${v.n} ${plural(v.n, ['колонка', 'колонки', 'колонок'])}`,
    apply: 'Применить',
    delete: 'Удалить',
    applying_position: (v) => `Применяю «${v.name}»…`,
    applied_with_problems: (v) => `Применено (${v.n}), но: ${v.problems}`,
    applied: (v) => `Позиция «${v.name}» применена, изменено колонок: ${v.n}.`,
    apply_failed: 'Не удалось применить: ',
    saved: (v) => `Позиция «${v.name}» сохранена.`,
    save_failed: 'Не сохранилось: ',
    no_microphone: 'Нет доступа к микрофону: ',
  },
};

function t(key, vars) {
  const text = STRINGS[LANG][key] ?? STRINGS.en[key] ?? key;
  return typeof text === 'function' ? text(vars || {}) : text;
}

// Static text in the page carries a key rather than words, so both languages
// live in one place.
function translatePage() {
  document.querySelectorAll('[data-i18n]').forEach((node) => {
    node.textContent = t(node.dataset.i18n);
  });
  document.querySelectorAll('[data-i18n-placeholder]').forEach((node) => {
    node.placeholder = t(node.dataset.i18nPlaceholder);
  });
}

// Every URL is relative to the page, never to the host root: behind Home
// Assistant's ingress the app lives under /api/hassio_ingress/<token>/, and an
// absolute '/api/…' would land on Home Assistant's own API instead.
const here = (path) => new URL(path, location.href).href;

// Match the page's own scheme: wss behind the TLS-terminating proxy, ws when
// developing against http://localhost (which browsers count as a secure
// context, so the microphone still works there).
const socketUrl = here('ws').replace(/^http/, 'ws');

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
  el('check').disabled = busy || !runnable;
  el('listen').disabled = busy || !runnable;
  el('refresh').disabled = busy;
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
  await audio.audioWorklet.addModule('static/recorder-worklet.js');

  node = new AudioWorkletNode(audio, 'speaker-sync-recorder');
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
    setStatus(t('disconnected'), 'err');
    setBusy(true);
    setTimeout(connect, 2000);
  };
  socket.onmessage = (event) => handle(JSON.parse(event.data));
}

function handle(message) {
  switch (message.type) {
    case 'record_start':
      recording = true;
      setStatus(t('recording'), 'live');
      break;

    case 'record_stop':
      recording = false;
      socket.send(JSON.stringify({
        type: 'recording_done',
        sample_rate: audio ? audio.sampleRate : 48000,
      }));
      setStatus(t('computing'), 'live');
      break;

    case 'progress':
      onProgress(message);
      break;

    case 'sign':
      setStatus(message.detail, message.conclusive ? 'ok' : 'err');
      setBusy(false);
      refreshPlayers();
      break;

    case 'check':
      showCheck(message);
      setBusy(false);
      refreshPlayers();
      break;

    case 'listening':
      setStatus(t('listening'), 'ok');
      setBusy(false);
      break;

    case 'report':
      showReport(message);
      setBusy(false);
      refreshPlayers();
      // A result now exists, so saving it as a position becomes possible.
      refreshProfiles();
      break;

    case 'error':
      setStatus(t('error') + message.message, 'err');
      setBusy(false);
      break;
  }
}

let passOffset = 0;
let passSpan = 1;

function onProgress(message) {
  if (message.stage === 'pass') {
    // Two passes either way: measure then prove, or one per track format.
    const second = message.which === 'after' || message.rate === 48000;
    passOffset = second ? 0.5 : 0;
    passSpan = 0.5;
    setStatus(t(message.which === 'after' ? 'verifying' : 'measuring'), 'live');
  } else if (message.stage === 'round') {
    setProgress(passOffset + passSpan * (message.round / message.rounds));
    setStatus(t('playing', {
      name: nameOf(message.player_id), round: message.round, rounds: message.rounds,
    }), 'live');
  } else if (message.stage === 'applying') {
    setStatus(t('applying', { writes: message.writes }), 'live');
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
    el('chirps-note').textContent = t('guard_too_many', { guard: session.guard_chirps });
    return;
  }

  // One round per speaker, plus a repeat of the first to measure clock drift,
  // and the whole thing runs twice: measure, then verify — the verification
  // with one more round at the end, every speaker at once.
  const rounds = ready ? ready + 1 : 0;
  const total = rounds * chirps;
  const seconds = Math.round((2 * total + (rounds ? chirps : 0)) * session.period_seconds);

  let note = t('readings', { n: readings });
  if (rounds) {
    // Spelled out because the first speaker sounding twice is otherwise a
    // surprise: at five per round it emits ten, and the count looks stuck.
    note += t('total', { total, seconds });
  }
  el('chirps-note').textContent = note;
}

// ------------------------------------------------------------------ players

let players = [];

function nameOf(playerId) {
  if (playerId === '*') return t('everyone');
  const found = players.find((p) => p.player_id === playerId);
  return found ? found.name : playerId;
}

// While Music Assistant is out of reach the server says why with a 503; the
// list is asked for again until it answers.
const RECONNECT_POLL_MS = 5000;
let reconnectTimer = null;

async function refreshPlayers() {
  clearTimeout(reconnectTimer);
  const response = await fetch('api/players');
  if (!response.ok) {
    players = [];
    runnable = false;
    refreshControls();
    el('players').innerHTML = '';
    setStatus((await response.text()).trim() || response.statusText, 'err');
    if (response.status === 503) {
      reconnectTimer = setTimeout(refreshPlayers, RECONNECT_POLL_MS);
    }
    return;
  }
  const data = await response.json();
  players = data.players;
  session = data;

  const chirps = el('chirps');
  chirps.min = data.chirps_range[0];
  chirps.max = data.chirps_range[1];
  // Only while the field is not being edited, or typing would fight the poll.
  if (document.activeElement !== chirps) chirps.value = data.chirps_per_round;

  // Re-rendering would fold every open settings list back up.
  const unfolded = new Set(
    [...document.querySelectorAll('#players details[open]')].map((d) => d.dataset.player),
  );

  const rows = players.map((p) => `
    <tr class="${p.enabled ? '' : 'off'}">
      <td>${escapeHtml(p.name)}<br><span class="sub">${escapeHtml(p.transport)}</span>${
        // Always folded away: the names are for diagnosis, not for reading.
        // The summary says so when they matter — the delay setting was not
        // found, and seeing what is there turns that into a one-line fix.
        p.config_keys && p.config_keys.length
          ? `<details class="keys" data-player="${escapeAttr(p.player_id)}"${
              unfolded.has(p.player_id) ? ' open' : ''}><summary>${t(
              p.enabled && p.available && !p.sync_adjust_key ? 'settings_found' : 'settings_all'
            )}</summary>${escapeHtml(p.config_keys.join(', '))}</details>`
          : ''
      }${
        // A failed read is a different fault from an absent setting, and the
        // server's own words are what identify it.
        p.config_error
          ? `<br><span class="sub bad">${escapeHtml(p.config_error)}</span>`
          : ''
      }</td>
      <td class="num">${p.sync_adjust_key
        ? `${p.sync_adjust_ms > 0 ? '+' : ''}${p.sync_adjust_ms} ${t('ms')}`
        : '—'}</td>
      <td>
        <span class="pill ${p.calibratable ? 'on' : 'off'}">${
          escapeHtml(p.calibratable ? t('ready_pill') : (p.excluded_because || t('not_taking_part')))
        }</span>
        <button class="toggle" data-toggle="${escapeAttr(p.player_id)}"
                data-enable="${p.enabled ? '0' : '1'}"${busy ? ' disabled' : ''}>${
          t(p.enabled ? 'switch_off' : 'switch_on')
        }</button>
      </td>
    </tr>`).join('');

  el('players').innerHTML =
    `<thead><tr><th>${t('speaker')}</th><th>sync_adjust</th><th></th></tr></thead>
     <tbody>${rows || `<tr><td colspan="3">${t('no_players')}</td></tr>`}</tbody>`;

  const ready = players.filter((p) => p.calibratable);
  runnable = ready.length >= 2;
  refreshControls();
  describeRoundLength();

  if (!runnable) {
    setStatus(
      t(players.length ? 'need_two' : 'no_players_status'),
      'err',
    );
  } else if (ready.some((p) => !p.is_sendspin)) {
    // Sendspin guarantees the tightest playback sync; other providers still
    // work, since any offset they add is measured and corrected, but the
    // result is only as steady as their own synchronisation.
    setStatus(t('ready_mixed', { n: ready.length }));
  } else {
    setStatus(t('ready', { n: ready.length }));
  }
}

async function setPlayerEnabled(playerId, enabled) {
  try {
    const response = await fetch(`api/players/${encodeURIComponent(playerId)}/enabled`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled }),
    });
    if (!response.ok) throw new Error((await response.text()).trim());
  } catch (error) {
    setStatus(t('toggle_failed') + error.message, 'err');
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
    `<thead><tr><th>${t('speaker')}</th><th>${t('was')}</th><th>${t('now')}</th>`
    + `<th>${t('residual')}</th></tr></thead><tbody>` +
    report.players.map((p) => `
      <tr>
        <td>${escapeHtml(p.name)}${p.clamped ? ` <span class="pill off">${t('limit')}</span>` : ''}</td>
        <td class="num">${p.current_adjust_ms} ${t('ms')}</td>
        <td class="num"><b>${p.target_adjust_ms} ${t('ms')}</b></td>
        <td class="num">${p.residual_error_ms.toFixed(1)} ${t('ms')}</td>
      </tr>`).join('') + '</tbody>';

  const notes = [];
  if (report.strategy === 'centered') {
    notes.push(['warn', t('centered')]);
  }
  if (report.strategy === 'align_to_fastest') {
    notes.push(['', t('fastest')]);
  }
  if (!report.fits) {
    notes.push(['bad', t('does_not_fit')]);
  }
  report.problems.forEach((p) => notes.push(['bad', p]));
  if (!notes.length && report.improved) {
    notes.push(['', t('verified')]);
  }
  if (report.together && report.together.confirmed) {
    notes.push(['', t('together_ok', { spread: report.together.spread_ms.toFixed(1) })]);
  }

  el('notes').innerHTML = notes
    .map(([kind, text]) => `<p class="note ${kind}">${escapeHtml(text)}</p>`)
    .join('');

  setStatus(t(report.improved ? 'done' : 'done_with_notes'),
            report.improved ? 'ok' : 'err');
}

function showCheck(check) {
  el('check-result').classList.remove('hidden');
  setProgress(1);

  const spread = el('check-spread');
  spread.textContent = check.spread_ms.toFixed(check.spread_ms < 10 ? 1 : 0);
  spread.className = 'headline ' + (check.in_sync ? 'good' : 'bad');

  const khz = (rate) => `${(rate / 1000).toLocaleString(LANG)} ${LANG === 'ru' ? 'кГц' : 'kHz'}`;
  el('check-table').innerHTML =
    `<thead><tr><th>${t('speaker')}</th>`
    + check.rates.map((r) => `<th>${khz(r)}</th>`).join('')
    + `</tr></thead><tbody>` +
    check.players.map((p) => `
      <tr>
        <td>${escapeHtml(p.name)}</td>
        ${p.offsets_ms.map((v) => `<td class="num">${
          v === null ? '—' : `+${v.toFixed(1)} ${t('ms')}`}</td>`).join('')}
      </tr>`).join('') + '</tbody>'
    + `<caption class="sub">${t('offset_from_first')}</caption>`;

  const notes = check.problems.map((p) => ['bad', p]);
  if (check.in_sync) notes.push(['', t('in_sync')]);
  if (check.together && check.together.confirmed) {
    notes.push(['', t('together_ok', { spread: check.together.spread_ms.toFixed(1) })]);
  }
  el('check-notes').innerHTML = notes
    .map(([kind, text]) => `<p class="note ${kind}">${escapeHtml(text)}</p>`)
    .join('');

  setStatus(t(check.in_sync ? 'done' : 'done_with_notes'), check.in_sync ? 'ok' : 'err');
}

async function listen() {
  setBusy(true);
  setStatus(t('listening'), 'live');
  socket.send(JSON.stringify({ type: 'listen' }));
}

// ----------------------------------------------------------------- profiles

async function refreshProfiles() {
  const response = await fetch('api/profiles');
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
      `<tbody><tr><td>${t('nothing_saved')}</td></tr></tbody>`;
    return;
  }

  el('profiles').innerHTML = '<tbody>' + data.profiles.map((p) => `
    <tr>
      <td>${escapeHtml(p.name)}<br><span class="pill">${t('n_speakers', { n: p.speakers })}</span></td>
      <td class="actions">
        <button data-apply="${escapeAttr(p.name)}">${t('apply')}</button>
        <button class="link" data-delete="${escapeAttr(p.name)}">${t('delete')}</button>
      </td>
    </tr>`).join('') + '</tbody>';
}

async function applyProfile(name) {
  setBusy(true);
  setStatus(t('applying_position', { name }), 'live');
  try {
    const response = await fetch(`api/profiles/${encodeURIComponent(name)}/apply`, {
      method: 'POST',
    });
    if (!response.ok) throw new Error(await response.text());

    const result = await response.json();
    const written = result.applied.length;
    setStatus(
      result.problems.length
        ? t('applied_with_problems', { n: written, problems: result.problems.join('; ') })
        : t('applied', { name, n: written }),
      result.problems.length ? 'err' : 'ok',
    );
  } catch (error) {
    setStatus(t('apply_failed') + error.message, 'err');
  } finally {
    setBusy(false);
    refreshPlayers();
  }
}

async function saveProfile() {
  const name = el('save-name').value.trim();
  if (!name) return;

  const response = await fetch('api/profiles', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  });
  if (response.ok) {
    el('save-name').value = '';
    setStatus(t('saved', { name }), 'ok');
    refreshProfiles();
  } else {
    setStatus(t('save_failed') + (await response.text()), 'err');
  }
}

async function deleteProfile(name) {
  await fetch(`api/profiles/${encodeURIComponent(name)}`, { method: 'DELETE' });
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
    setStatus(t('no_microphone') + error.message, 'err');
    // Inside Home Assistant's panel the page is an iframe, and whether it may
    // ask for the microphone is the embedding page's call, not ours. The same
    // ingress address works as a top-level tab, where the browser asks directly.
    if (window.top !== window.self) el('popout').hidden = false;
    setBusy(false);
  }
}

el('chirps').addEventListener('input', describeRoundLength);
el('run').addEventListener('click', () => start('calibrate'));
el('check').addEventListener('click', () => start('check'));
el('listen').addEventListener('click', listen);
el('save').addEventListener('click', saveProfile);
el('refresh').addEventListener('click', refreshPlayers);

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

translatePage();
refreshPlayers();
refreshProfiles();
connect();
