/**
 * Small labelled evaluation set for the smart-home desktop-assistant domain used throughout this PoC.
 *
 * Labels were written by hand for this PoC and are intentionally clear-cut (a calibration set must have
 * trustworthy gold labels). Replace / extend with real traffic from your workflow before relying on the
 * fitted temperatures: ~60 examples give a usable but noisy estimate (see the LOO numbers in the report).
 *
 * Fields: message -> the user's utterance; intent / target_device -> gold option keys of the questions in
 * src/questions.mjs; should_execute -> gold boolean for the noul question.
 */
import { STATE } from "../src/questions.mjs";

const c = (message, target_device) => ({ message, intent: "control_device", should_execute: true, target_device });
const a = (message, target_device = "none") => ({ message, intent: "ask_question", should_execute: false, target_device });
const t = (message) => ({ message, intent: "chat", should_execute: false, target_device: "none" });
const u = (message) => ({ message, intent: "unknown", should_execute: false, target_device: "none" });

export const EVAL_SET = [
  // ---- control_device -------------------------------------------------------------------------------
  c("Turn off the living room lights", "lights"),
  c("Switch on the kitchen lights please", "lights"),
  c("Dim the bedroom lights to 30 percent", "lights"),
  c("Lights off everywhere, I'm going to bed", "lights"),
  c("Can you turn on the porch light?", "lights"),
  c("Turn off all the lights in the house", "lights"),
  c("Set the living room lights to warm white", "lights"),
  c("Bedroom lights on", "lights"),
  c("Kill the lights", "lights"),
  c("Brighten the office lights", "lights"),
  c("Turn the lights blue in the living room", "lights"),
  c("Set the thermostat to 21 degrees", "thermostat"),
  c("Make it warmer in here", "thermostat"),
  c("Turn the heating down a bit", "thermostat"),
  c("Would you mind switching the AC to cool mode?", "thermostat"),
  c("Raise the temperature to 23", "thermostat"),
  c("Turn off the thermostat", "thermostat"),
  c("Cool the house down to 19 degrees", "thermostat"),
  c("Lock the front door", "locks"),
  c("Unlock the back door, I'm coming in with groceries", "locks"),
  c("Lock all the doors, we're leaving", "locks"),
  c("Lock the garage door", "locks"),
  c("Play some jazz on the living room speaker", "media"),
  c("Pause the TV", "media"),
  c("Turn the volume down", "media"),
  c("Skip this song", "media"),
  c("Mute the TV", "media"),
  c("Please stop the music", "media"),
  c("Put on the news on the TV", "media"),
  c("Turn on the TV", "media"),
  c("Next track", "media"),
  // ---- ask_question ---------------------------------------------------------------------------------
  a("Are the living room lights still on?", "lights"),
  a("Which lights are on right now?", "lights"),
  a("Did I leave the garage lights on?", "lights"),
  a("What's the temperature in the bedroom?", "thermostat"),
  a("How much energy did the heating use this month?", "thermostat"),
  a("How do I change the thermostat schedule?", "thermostat"),
  a("Is the front door locked?", "locks"),
  a("When did I last lock the door?", "locks"),
  a("Is the back door open?", "locks"),
  a("What song is playing right now?", "media"),
  a("What's the volume set to?", "media"),
  a("What time is it?"),
  a("What's the weather like tomorrow?"),
  a("What's on my calendar today?"),
  a("Is it going to rain this weekend?"),
  a("How many devices are connected?"),
  // ---- chat -----------------------------------------------------------------------------------------
  t("Hey, how are you doing today?"),
  t("Good morning!"),
  t("Tell me a joke"),
  t("Thanks, that was helpful"),
  t("You're pretty smart for a computer"),
  t("I had a really long day"),
  t("Goodnight"),
  t("Haha, that's funny"),
  t("I'm so bored"),
  t("Nice to meet you"),
  // ---- unknown --------------------------------------------------------------------------------------
  u("Do the thing"),
  u("asdf qwerty"),
  u("The thing with the, you know"),
  u("Okay so"),
  u("Yes"),
  u("Um"),
  u("purple elephants dancing on tuesday"),
  u("make it do that again but different"),
];

/** Build the state object for one example (same template as the PoC). */
export const stateFor = (ex) => ({ ...STATE, userMessage: ex.message });
