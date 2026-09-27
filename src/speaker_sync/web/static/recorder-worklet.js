// Raw microphone capture.
//
// MediaRecorder is deliberately not used: it hands back Opus or similar, and
// the encoder's delay is both unknown and variable. This whole app measures
// timing, so the audio has to stay as the untouched float samples the device
// produced.
//
// Frames arrive 128 samples at a time. Posting each one would mean ~375
// messages a second, so they are gathered into larger blocks first. The input
// buffer is reused by the engine between calls, hence the copy.

const BLOCK_SIZE = 4096;

class RecorderProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(BLOCK_SIZE);
    this.filled = 0;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) {
      // No input yet; keep the processor alive rather than ending the stream.
      return true;
    }

    let offset = 0;
    while (offset < channel.length) {
      const take = Math.min(BLOCK_SIZE - this.filled, channel.length - offset);
      this.buffer.set(channel.subarray(offset, offset + take), this.filled);
      this.filled += take;
      offset += take;

      if (this.filled === BLOCK_SIZE) {
        this.port.postMessage(this.buffer.slice());
        this.filled = 0;
      }
    }

    return true;
  }
}

registerProcessor('speaker-sync-recorder', RecorderProcessor);
