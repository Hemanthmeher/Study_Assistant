const express = require('express');
const multer = require('multer');
const path = require('path');
const dotenv = require('dotenv');
const mammoth = require('mammoth');
const pdfParse = require('pdf-parse');
const Groq = require('groq-sdk');
const { pipeline } = require('@xenova/transformers');
const { v4: uuidv4 } = require('uuid');
const Tesseract = require('tesseract.js'); // For OCR
const { ChromaClient } = require('chromadb'); // Added ChromaDB

dotenv.config();

const app = express();
const PORT = 3000;

app.use(express.json());

// Configure Groq API
const groq = new Groq({
  apiKey: process.env.GROQ_API_KEY
});

// Configure ChromaDB
const chroma = new ChromaClient({ path: "http://localhost:8000" });
let collection = null;

// Initialize Chroma Collection
async function initChroma() {
  try {
    console.log('📡 Connecting to ChromaDB (localhost:8000)...');
    collection = await chroma.getOrCreateCollection({
      name: "study_assistant_collection",
      embeddingFunction: null, // CRITICAL: Tells Chroma we handle embeddings manually
      metadata: { "hnsw:space": "cosine" }
    });
    console.log('✅ ChromaDB Collection "study_assistant_collection" ready!');
  } catch (err) {
    console.error('❌ ChromaDB Connection Failed. Make sure "chroma run" is active in a separate terminal.');
    console.error('Error Details:', err.message);
  }
}
initChroma();

// Allowed file types
const ALLOWED_MIME_TYPES = [
  'application/pdf',
  'application/msword',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  'text/plain',
  'image/png',
  'image/jpeg'
];

const MAX_FILE_SIZE = 30 * 1024 * 1024;
const storage = multer.memoryStorage();
const upload = multer({
  storage,
  limits: { fileSize: MAX_FILE_SIZE },
  fileFilter: (req, file, cb) => {
    if (ALLOWED_MIME_TYPES.includes(file.mimetype)) cb(null, true);
    else cb(new Error('Invalid file type'));
  },
});

app.use(express.static(path.join(__dirname, 'public')));

// ─── Helper: Convert Buffer to Text ─────────────────────────────────
async function extractTextFromFile(file) {
  const mime = file.mimetype;
  if (mime === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document') {
    const result = await mammoth.extractRawText({ buffer: file.buffer });
    return result.value;
  }
  if (mime === 'application/pdf') {
    const data = await pdfParse(file.buffer);
    return data.text;
  }
  if (mime === 'text/plain') {
    return file.buffer.toString('utf-8');
  }
  if (mime === 'image/png' || mime === 'image/jpeg') {
    console.log('🔍 Running OCR on image using Tesseract.js...');
    const worker = await Tesseract.createWorker('eng');
    const ret = await worker.recognize(file.buffer);
    await worker.terminate();
    console.log('✅ OCR extraction successful!');
    return ret.data.text;
  }
  return '';
}

// ─── Helper: Groq summarization ─────────────────────────────────────
async function summarizeWithGroq(text, format) {
  try {
    let userPrompt = '';
    if (format === 'paragraphs') userPrompt = 'Summarize in a concise paragraph:\n\n' + text.slice(0, 15000);
    else if (format === 'bullets') userPrompt = 'Summarize as clear bullet points:\n\n' + text.slice(0, 15000);
    else if (format === 'detailed') userPrompt = 'Give a detailed summary of:\n\n' + text.slice(0, 15000);
    else userPrompt = 'Summarize this text:\n\n' + text.slice(0, 15000);

    const chatCompletion = await groq.chat.completions.create({
      messages: [
        { role: "system", content: "You are a helpful assistant that creates clear and concise summaries." },
        { role: "user", content: userPrompt }
      ],
      model: "llama-3.3-70b-versatile",
      temperature: 0.5,
      max_tokens: 2048,
    });
    return chatCompletion.choices[0]?.message?.content || "No summary generated";
  } catch (error) {
    console.error('❌ Groq API Error:', error.message);
    throw new Error(`Groq API Error: ${error.message}`);
  }
}

// ─── RAG: Chunking helper ───────────────────────────────────────────
function chunkText(text, chunkSize = 500, overlap = 50) {
  const words = text.split(/\s+/);
  const chunks = [];
  for (let i = 0; i < words.length; i += chunkSize - overlap) {
    const chunk = words.slice(i, i + chunkSize).join(' ');
    if (chunk.trim().length > 0) chunks.push(chunk);
  }
  return chunks;
}

// ─── RAG: Embedding model ──────────────────────────────────────────
let embedder = null;
async function getEmbedder() {
  if (!embedder) {
    embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
  }
  return embedder;
}

// ─── Metadata storage (RAM for fast access to full text/list) ────────
const docMetadataStore = new Map(); // docId -> { filename, chunkCount, fullText, uploadedAt }

// ═══════════════════════════════════════════════════════════════════
// EXISTING ROUTE: Process file and summarize
// ═══════════════════════════════════════════════════════════════════
app.post('/api/process', upload.array('files', 10), async (req, res) => {
  try {
    const files = req.files;
    const format = req.body.format || 'bullets';
    if (!files || files.length === 0) return res.status(400).json({ error: 'No files uploaded' });

    let textContent = '';
    for (const f of files) textContent += await extractTextFromFile(f) + '\n\n';
    if (!textContent || textContent.trim().length === 0) return res.status(400).json({ error: 'No content found' });

    const summary = await summarizeWithGroq(textContent, format);
    return res.json({ summary, charCount: textContent.length });
  } catch (err) {
    console.error('❌ Error in /api/process:', err);
    return res.status(500).json({ error: err.message });
  }
});

// ═══════════════════════════════════════════════════════════════════
// FEATURE 1: RAG — Upload to ChromaDB
// ═══════════════════════════════════════════════════════════════════
app.post('/api/upload', upload.array('files', 10), async (req, res) => {
  try {
    if (!collection) return res.status(503).json({ error: 'ChromaDB not ready. Run "chroma run" in terminal.' });

    const files = req.files;
    if (!files || files.length === 0) return res.status(400).json({ error: 'No files uploaded' });

    let textContent = '';
    let names = [];
    for (const f of files) {
      names.push(f.originalname);
      textContent += await extractTextFromFile(f) + '\n\n';
    }

    if (!textContent || textContent.trim().length === 0) return res.status(400).json({ error: 'Empty file contents' });

    const chunks = chunkText(textContent, 500, 50);
    const docId = uuidv4();
    const embed = await getEmbedder();

    const ids = [];
    const embeddings = [];
    const metadatas = [];
    const documents = [];

    console.log(`🧠 Embedding ${chunks.length} chunks for ChromaDB...`);

    for (let i = 0; i < chunks.length; i++) {
      const output = await embed(chunks[i], { pooling: 'mean', normalize: true });
      embeddings.push(Array.from(output.data));
      ids.push(`${docId}_chunk_${i}`);
      metadatas.push({ doc_id: docId, filename: names.join(', ') });
      documents.push(chunks[i]);
    }

    // ADD TO CHROMA
    await collection.add({ ids, embeddings, metadatas, documents });

    // Store simple metadata in RAM for list view & mindmaps
    docMetadataStore.set(docId, {
      filename: names.join(', '),
      chunkCount: chunks.length,
      fullText: textContent,
      uploadedAt: new Date().toISOString()
    });

    console.log(`✅ ChromaDB Stored: ${docId} (${chunks.length} chunks)`);
    return res.json({ success: true, doc_id: docId, filename: names.join(', '), chunks_stored: chunks.length });

  } catch (err) {
    console.error('❌ Error in /api/upload:', err);
    return res.status(500).json({ error: 'ChromaDB Upload Error: ' + err.message });
  }
});

// ═══════════════════════════════════════════════════════════════════
// FEATURE 1: RAG — Chat with ChromaDB
// ═══════════════════════════════════════════════════════════════════
app.post('/api/chat', async (req, res) => {
  try {
    if (!collection) return res.status(503).json({ error: 'ChromaDB not ready.' });
    const { question, doc_id } = req.body;
    if (!question || !doc_id) return res.status(400).json({ error: 'Missing input' });

    console.log(`💬 Chroma Query: "${question}" on doc ${doc_id}`);

    // Embed question
    const embed = await getEmbedder();
    const qOutput = await embed(question, { pooling: 'mean', normalize: true });
    const qEmbedding = Array.from(qOutput.data);

    // QUERY CHROMA
    const results = await collection.query({
      queryEmbeddings: [qEmbedding],
      nResults: 5,
      where: { "doc_id": doc_id }
    });

    const context = results.documents[0].join('\n\n---\n\n');
    console.log(`🔍 Retrieved ${results.documents[0].length} relevant chunks`);

    const chatCompletion = await groq.chat.completions.create({
      messages: [
        { role: "system", content: "You are a helpful study assistant. Answer using ONLY context below.\n\nContext:\n" + context },
        { role: "user", content: question }
      ],
      model: "llama-3.3-70b-versatile",
      temperature: 0.3,
      max_tokens: 1024,
    });

    const answer = chatCompletion.choices[0]?.message?.content || "No answer";
    const sources = results.documents[0].slice(0, 3).map((text, i) => ({
      excerpt: text.substring(0, 150) + '...',
      score: Math.round((1 - results.distances[0][i]) * 100) / 100
    }));

    return res.json({ answer, sources });
  } catch (err) {
    console.error('❌ Error in /api/chat:', err);
    return res.status(500).json({ error: err.message });
  }
});

app.get('/api/documents', (req, res) => {
  const docs = [];
  docMetadataStore.forEach((meta, id) => {
    docs.push({ doc_id: id, filename: meta.filename, chunk_count: meta.chunkCount, uploaded_at: meta.uploadedAt });
  });
  return res.json(docs);
});

// ═══════════════════════════════════════════════════════════════════
// FEATURE 2: Mind Map Generation
// ═══════════════════════════════════════════════════════════════════
app.post('/api/mindmap', upload.array('files', 10), async (req, res) => {
  try {
    const files = req.files;
    if (!files || files.length === 0) return res.status(400).json({ error: 'No files' });

    let textContent = '';
    let names = [];
    for (const f of files) {
      names.push(f.originalname);
      textContent += await extractTextFromFile(f) + '\n\n';
    }

    console.log(`🗺️ Mind Map: ${names.join(', ')}`);

    const chatCompletion = await groq.chat.completions.create({
      messages: [
        {
          role: "system",
          content: `You are a professional mind map generator. Analyze the document and return a detailed hierarchical structure in valid JSON format.
Exact Schema:
{
  "title": "Main Central Topic",
  "children": [
    {
      "label": "Main Branch 1",
      "children": [
        { "label": "Sub-concept 1.1" },
        { "label": "Sub-concept 1.2" }
      ]
    },
    {
      "label": "Main Branch 2",
      "children": [
        { "label": "Sub-concept 2.1" }
      ]
    }
  ]
}
Requirements:
1. Generate 4-7 main branches.
2. Each main branch must have 2-4 sub-children.
3. Keep labels very concise (max 5 words).
4. Return ONLY the JSON object.`
        },
        {
          role: "user",
          content: "Generate a detailed and logically structured mind map for this document:\n\n" + textContent.slice(0, 12000)
        }
      ],
      model: "llama-3.3-70b-versatile",
      response_format: { type: "json_object" }
    });

    const mindmapData = JSON.parse(chatCompletion.choices[0].message.content);
    const docId = uuidv4();
    docMetadataStore.set(docId, { filename: names.join(', '), fullText: textContent, uploadedAt: new Date().toISOString(), chunkCount: 0 });
    mindmapData.doc_id = docId;

    return res.json(mindmapData);
  } catch (err) {
    console.error('❌ Mindmap Error:', err);
    return res.status(500).json({ error: err.message });
  }
});

app.post('/api/node-details', async (req, res) => {
  try {
    const { label, doc_id } = req.body;
    const doc = docMetadataStore.get(doc_id);
    if (!doc) return res.status(404).json({ error: 'Context not found.' });

    const chatCompletion = await groq.chat.completions.create({
      messages: [
        { role: "system", content: "Explain simply in 2-3 sentences based ONLY on context." },
        { role: "user", content: `Context:\n${doc.fullText.slice(0, 12000)}\n\nExplain: ${label}` }
      ],
      model: "llama-3.3-70b-versatile",
    });
    return res.json({ explanation: chatCompletion.choices[0].message.content });
  } catch (err) {
    return res.status(500).json({ error: err.message });
  }
});

app.get('/api/health', (req, res) => res.json({ status: 'ok', chroma: !!collection }));

// Start server
app.listen(PORT, () => {
  console.log(`✅ Server running at http://localhost:${PORT}`);
  console.log(`🚀 RAG Persistence: ChromaDB`);
  console.log(`💡 REMINDER: Run "chroma run" in a separate terminal to enable the Vector Database.`);
  getEmbedder().catch(err => console.warn('⚠️ Embedding model pre-load failed:', err.message));
});