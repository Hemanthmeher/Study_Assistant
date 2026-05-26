// test-grok-models.js
const OpenAI = require('openai');
require('dotenv').config();

const xai = new OpenAI({
  apiKey: process.env.XAI_API_KEY,
  baseURL: "https://api.x.ai/v1"
});

const modelsToTest = [
  'grok-beta',
  'grok-2-latest',
  'grok-2-1212',
  'grok-vision-beta'
];

async function testModel(modelName) {
  try {
    console.log(`\nTesting: ${modelName}...`);
    
    const completion = await xai.chat.completions.create({
      model: modelName,
      messages: [
        {
          role: "system",
          content: "You are Grok, a chatbot inspired by the Hitchhiker's Guide to the Galaxy."
        },
        {
          role: "user",
          content: "Say hello in one word"
        }
      ],
      temperature: 0
    });

    const response = completion.choices[0].message.content;
    console.log(`✅ ${modelName} WORKS!`);
    console.log(`   Response: ${response}`);
    return true;
    
  } catch (error) {
    console.log(`❌ ${modelName} failed: ${error.message}`);
    return false;
  }
}

async function findWorkingModel() {
  console.log('🔍 Testing available Grok models with your API key...\n');
  
  for (const modelName of modelsToTest) {
    const works = await testModel(modelName);
    if (works) {
      console.log(`\n✨ USE THIS MODEL: "${modelName}"\n`);
    }
    // Wait a bit between requests
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
  
  console.log('\n✅ Testing complete!');
}

findWorkingModel();