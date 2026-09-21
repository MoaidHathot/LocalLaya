/**
 * Shared state + question sets for the PoC and benchmarks.
 * The 3-question set is exactly the one from the original plan, so the results are directly comparable.
 * Each question's options must fit in head_max_len (192 tokens); the state is truncated to 512 tokens.
 */

export const STATE = {
  application: "Windows desktop assistant",
  userMessage: "Turn off the living room lights",
  time: "23:30",
  livingRoomLights: "on",
};

const intent = {
  type: "choice",
  instructions: "What does the user want to do?",
  criteria: {
    control_device: "Control a smart home device",
    ask_question: "Ask for information",
    chat: "General conversation",
    unknown: "Intent is unclear",
  },
};

const should_execute = {
  type: "noul",
  instructions: "Should the assistant execute a smart-home action based on this request?",
};

const urgency = {
  type: "score",
  instructions: "How urgent is this request?",
  criteria: ["not urgent", "somewhat urgent", "urgent", "critical"],
};

export const QUESTIONS_1 = { intent };

export const QUESTIONS_3 = { intent, should_execute, urgency };

export const QUESTIONS_10 = {
  intent,
  should_execute,
  urgency,
  target_device: {
    type: "choice",
    instructions: "Which device does the user refer to?",
    criteria: {
      lights: "Lighting",
      thermostat: "Heating or cooling",
      media: "TV, speakers, music",
      locks: "Doors and locks",
      none: "No device mentioned",
    },
  },
  desired_state: {
    type: "choice",
    instructions: "What state does the user want the device in?",
    criteria: { on: "Turn on / activate", off: "Turn off / deactivate", adjust: "Change a level or setting", unknown: "Not specified" },
  },
  room: {
    type: "choice",
    instructions: "Which room is referenced?",
    criteria: ["living room", "bedroom", "kitchen", "bathroom", "office", "unspecified"],
  },
  needs_confirmation: {
    type: "noul",
    instructions: "Should the assistant ask the user to confirm before acting?",
  },
  is_ambiguous: {
    type: "noul",
    instructions: "Is the request ambiguous or missing required details?",
  },
  sentiment: {
    type: "score",
    instructions: "What is the user's tone?",
    criteria: ["angry", "frustrated", "neutral", "friendly"],
  },
  complexity: {
    type: "score",
    instructions: "How complex is this request to fulfil?",
    criteria: ["trivial", "simple", "moderate", "complex", "very complex"],
  },
};

export const QUESTION_SETS = { 1: QUESTIONS_1, 3: QUESTIONS_3, 10: QUESTIONS_10 };
