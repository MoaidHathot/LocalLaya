/** Public question aliases -> the canonical schema understood by every model backend. */
import { isDeepStrictEqual } from "node:util";

const object = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
const bad = (message) => Object.assign(new Error(message), { status: 400, code: "BAD_REQUEST" });

/** Validate without mutating input. Already-canonical question sets are returned as-is. */
export function normalizeQuestions(questions) {
  if (!object(questions) || !Object.keys(questions).length) throw bad("questions must be a non-empty object");
  let changed = false;
  const entries = Object.entries(questions).map(([id, question]) => {
    const type = question?.type === "predicate" ? "noul" : question?.type;
    if (!object(question) || !["choice", "score", "noul"].includes(type)) throw bad(`question ${id}: type must be choice | score | noul | predicate`);
    if (question.instructions === undefined) throw bad(`question ${id}: instructions required`);

    const hasChoices = Object.hasOwn(question, "choices");
    const hasLevels = Object.hasOwn(question, "levels");
    if (hasChoices && type !== "choice") throw bad(`question ${id}: choices is only valid for type=choice`);
    if (hasLevels && type !== "score") throw bad(`question ${id}: levels is only valid for type=score`);
    const alias = hasChoices ? "choices" : hasLevels ? "levels" : null;
    let criteria = question.criteria;
    if (alias) {
      if (criteria !== undefined && !isDeepStrictEqual(criteria, question[alias])) throw bad(`question ${id}: criteria and ${alias} conflict; provide one field or matching values`);
      if (criteria === undefined) criteria = question[alias];
    }
    if (type === "choice" && !(Array.isArray(criteria) ? criteria.length >= 2 : object(criteria) && Object.keys(criteria).length >= 2)) {
      throw bad(`question ${id}: choice needs >= 2 criteria (or choices)`);
    }
    if (type === "score" && !(Array.isArray(criteria) && criteria.length >= 2)) {
      throw bad(`question ${id}: score needs an ordered array of >= 2 levels in criteria or levels`);
    }

    if (!alias && type === question.type) return [id, question];
    changed = true;
    const normalized = { ...question, type };
    if (alias) {
      normalized.criteria = criteria;
      delete normalized[alias];
    }
    return [id, normalized];
  });
  return changed ? Object.fromEntries(entries) : questions;
}

/** Compatibility entry point for callers that only need validation. */
export function validateQuestions(questions) {
  normalizeQuestions(questions);
}
