import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const SIMILARITY_PCT = 70;
const PAGES_BONUS = 2;
const HUNDRED = 100;

const bibFile = path.join(__dirname, "paper", "references.bib");
const sectionsDir = path.join(__dirname, "paper", "sections");
const content = fs.readFileSync(bibFile, "utf-8");

// Parse entries
const entries = [];
const entryRegex = /(@\w+\{([^,]+),[\s\S]*?\n\})/g;
let m;
while ((m = entryRegex.exec(content)) !== null) {
    const full = m[1];
    const key = m[2].trim();
    const tm = full.match(/title\s*=\s*[{"](.+?)[}"]/s);
    const title = tm ? tm[1].replace(/[{}\s]+/g, " ").trim().toLowerCase() : "";
    const fields = (full.match(/^\s+\w+\s*=/gm) || []).length;
    const hasPages = /pages\s*=/.test(full);
    entries.push({ key, text: full, title, fields, hasPages });
}
console.log("Parsed " + entries.length + " entries");

// Remove wrong entries (Scholar returned genuinely wrong paper)
for (let i = entries.length - 1; i >= 0; i--) {
    if (entries[i].key === "myerson2023game" && entries[i].title.includes("first world war")) {
        console.log("REMOVING wrong: " + entries[i].key);
        entries.splice(i, 1);
    }
}

// Find duplicates by title word overlap
const seen = new Map();
const toRemove = new Set();
const keyMap = {};

for (const e of entries) {
    const words = new Set(e.title.replace(/[^a-z0-9\s]/g, "").split(/\s+/).filter(Boolean));
    let matched = false;
    for (const [st, se] of seen.entries()) {
        const sw = new Set(st.split(/\s+/).filter(Boolean));
        if (words.size > 0 && sw.size > 0) {
            let overlap = 0;
            for (const w of words) { if (sw.has(w)) overlap++; }
            if (overlap * HUNDRED > SIMILARITY_PCT * Math.min(words.size, sw.size)) {
                const sa = se.fields + (se.hasPages ? PAGES_BONUS : 0);
                const sb = e.fields + (e.hasPages ? PAGES_BONUS : 0);
                const [better, worse] = sb > sa ? [e, se] : [se, e];
                console.log("DUPLICATE: keep " + better.key + " (" + better.fields + "f), remove " + worse.key + " (" + worse.fields + "f)");
                toRemove.add(worse.key);
                if (worse.key !== better.key) keyMap[worse.key] = better.key;
                matched = true;
                break;
            }
        }
    }
    if (!matched) {
        seen.set(e.title.replace(/[^a-z0-9\s]/g, ""), e);
    }
}

const cleaned = entries.filter(e => !toRemove.has(e.key));

// Update tex cite keys in all .tex files
function findTexFiles(dir) {
    let files = [];
    for (const f of fs.readdirSync(dir, { withFileTypes: true })) {
        const fp = path.join(dir, f.name);
        if (f.isDirectory()) files = files.concat(findTexFiles(fp));
        else if (f.name.endsWith(".tex")) files.push(fp);
    }
    return files;
}

const texFiles = findTexFiles(sectionsDir);
for (const [oldKey, newKey] of Object.entries(keyMap)) {
    for (const tf of texFiles) {
        let c = fs.readFileSync(tf, "utf-8");
        const re = new RegExp("(\\\\cite[tp]?\\{[^}]*)" + oldKey.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + "\\b", "g");
        const nc = c.replace(re, "$1" + newKey);
        if (nc !== c) {
            fs.writeFileSync(tf, nc);
            console.log("  Updated " + oldKey + " -> " + newKey + " in " + path.basename(tf));
        }
    }
}

// Write cleaned bib
const out = cleaned.map(e => e.text).join("\n\n") + "\n";
fs.writeFileSync(bibFile, out);
console.log("\nResult: " + cleaned.length + " entries, " + out.split("\n").length + " lines");
console.log("Key mappings: " + JSON.stringify(keyMap));
