/**
 * Question wording variants for the three questions with gold labels in data/smart-home-eval.mjs.
 *
 * v1 = the wording used in the PoC / benchmarks (src/questions.mjs).
 * v2 = same options, more explicit criteria descriptions targeting the confusions seen in the v1 report
 *      (device questions classified as control; "volume"/"TV" not associated with media).
 *
 * Laya scores each option at its own [MASK] token from the option text, so wording IS the model input:
 * changing criteria descriptions is the cheapest accuracy lever, before fine-tuning.
 */
import { QUESTIONS_10 } from "../src/questions.mjs";

export const VARIANTS = {
  v1: {
    intent: QUESTIONS_10.intent,
    should_execute: QUESTIONS_10.should_execute,
    target_device: QUESTIONS_10.target_device,
  },
  v2: {
    intent: {
      type: "choice",
      instructions: "Classify the user's message. Is it a command that changes something, a question that only asks for information, small talk, or not understandable?",
      criteria: {
        control_device: "A command to change a device: turn on/off, set, dim, lock, unlock, play, pause, mute, adjust",
        ask_question: "A question asking for information or the current status; nothing should be changed",
        chat: "Small talk: greetings, thanks, jokes, feelings, opinions",
        unknown: "Gibberish, a fragment, or a request that cannot be understood",
      },
    },
    should_execute: {
      type: "noul",
      instructions: "Is this an explicit command to change a smart-home device, so that the assistant should execute an action now?",
      criteria: {
        true: "yes, the user clearly asked for a device action to be performed",
        false: "no: it is a question, small talk, or unclear, so nothing should be executed",
      },
    },
    target_device: {
      type: "choice",
      instructions: "Which kind of device does the user's message refer to (even when only asking about it)?",
      criteria: {
        lights: "Lights, lamps, brightness, light colour",
        thermostat: "Thermostat, heating, cooling, AC, temperature",
        media: "TV, speakers, music, songs, volume, playback, news",
        locks: "Doors, locks, garage door",
        none: "No device is mentioned",
      },
    },
  },
};

/** v3 = best measured wording per question (intent + should_execute from v2, target_device from v1). */
VARIANTS.v3 = { intent: VARIANTS.v2.intent, should_execute: VARIANTS.v2.should_execute, target_device: VARIANTS.v1.target_device };
