// One-off numeric diff: predictionMath.js (the JS port) vs the real
// Python backend, using the fixture written by
// scripts/_gen_prediction_math_fixture.py and the baselines written by
// scripts/export_static_data.py. Not part of the app or CI -- a manual
// check to run after editing either side of the port.
//
// Usage: node scripts/verify_prediction_math.mjs

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import {
  predictCompanyDetail, predictCompanyBulk, predictSector,
} from "../frontend/src/predictionMath.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const DATA = join(__dirname, "..", "frontend", "public", "data");

const readJson = (p) => JSON.parse(readFileSync(p, "utf-8"));

const fixture = readJson(join(DATA, "_verify_fixture.json"));
const companies = readJson(join(DATA, "stage6_predictions", "companies.json")).companies;
const sectors = readJson(join(DATA, "stage6_predictions", "sectors.json")).sectors;
const constants = readJson(join(DATA, "stage6_predictions", "constants.json"));

const companyByName = Object.fromEntries(companies.map((c) => [c.company, c]));
const sectorByName = Object.fromEntries(sectors.map((s) => [s.sector, s]));

const TOL = 0.05; // percentage points -- allows for rounding-order noise
let failures = 0;
let checks = 0;

function approx(a, b, label) {
  checks++;
  if (a == null && b == null) return;
  if (typeof a !== "number" || typeof b !== "number" || Math.abs(a - b) > TOL) {
    failures++;
    console.log(`  MISMATCH ${label}: js=${a} py=${b}`);
  }
}

function eq(a, b, label) {
  checks++;
  if (a !== b) {
    failures++;
    console.log(`  MISMATCH ${label}: js=${JSON.stringify(a)} py=${JSON.stringify(b)}`);
  }
}

console.log(`Checking ${fixture.detail.length} company-detail cases...`);
for (const { company, sliders, expected } of fixture.detail) {
  const baseline = companyByName[company];
  const got = predictCompanyDetail(baseline, sliders, constants.regimeMultiplier);
  eq(got.signal, expected.signal, `${company} detail signal @ ${JSON.stringify(sliders)}`);
  approx(got.confidence, expected.confidence, `${company} detail confidence`);
  approx(got.score, expected.score, `${company} detail score`);
  for (const h of [1, 5, 10]) {
    approx(got.predictions[h].return_pct, expected.predictions[h].return_pct, `${company} detail ${h}D return_pct`);
  }
}

console.log(`Checking ${fixture.bulk.length} company-bulk cases (15 companies each)...`);
for (const { sliders, expected } of fixture.bulk) {
  for (const row of expected) {
    const baseline = companyByName[row.Company];
    const got = predictCompanyBulk(baseline, sliders, constants.regimeMultiplier);
    eq(got.signal, row.Signal, `${row.Company} bulk signal @ ${JSON.stringify(sliders)}`);
    approx(got.confidence, row.Confidence, `${row.Company} bulk confidence`);
    approx(got.score, row.Score, `${row.Company} bulk score`);
    approx(got.predictions[5].return_pct, row["5D %"], `${row.Company} bulk 5D %`);
  }
}

console.log(`Checking ${fixture.sectors.length} sector cases (6 sectors each)...`);
for (const { sliders, expected } of fixture.sectors) {
  for (const row of expected) {
    const baseline = sectorByName[row.Sector];
    const got = predictSector(baseline, sliders, constants.regimeMultiplier);
    eq(got.signal, row.Signal, `${row.Sector} signal @ ${JSON.stringify(sliders)}`);
    approx(got.confidence, row.Conf, `${row.Sector} confidence`);
    approx(got.score, row.Score, `${row.Sector} score`);
    approx(got.predictions[5].return_pct, row["5D %"], `${row.Sector} 5D %`);
    approx(got.predictions[10].return_pct, row["10D %"], `${row.Sector} 10D %`);
  }
}

console.log(`\n${checks} checks, ${failures} mismatches.`);
process.exit(failures > 0 ? 1 : 0);
