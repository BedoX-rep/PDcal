import * as fs from "fs";
import { GoogleGenAI } from "@google/genai";

// Initialize Gemini with API key from environment
const ai = new GoogleGenAI({ apiKey: process.env.GOOGLE_API_KEY || "" });

export interface OcularHeightResult {
  leftOcularHeight: number;
  rightOcularHeight: number;
  confidence: number;
  analysisNotes: string;
}

// Validation function for ocular height measurements - only check for basic data validity
function validateOcularHeight(result: OcularHeightResult): boolean {
  const { leftOcularHeight, rightOcularHeight, confidence } = result;
  
  // Only check for valid numbers and reasonable confidence
  if (isNaN(leftOcularHeight) || isNaN(rightOcularHeight) || 
      leftOcularHeight <= 0 || rightOcularHeight <= 0) {
    return false;
  }
  
  // Basic confidence check
  if (confidence < 0.5 || confidence > 1.0) {
    return false;
  }
  
  return true;
}

export async function analyzeOcularHeight(imagePath: string, pdValue: number): Promise<OcularHeightResult> {
  const maxRetries = 3;
  let lastError: Error | null = null;
  
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const imageBytes = fs.readFileSync(imagePath);

      const prompt = `OPTICAL MEASUREMENT SYSTEM - OCULAR HEIGHT TO LENS EDGE

CRITICAL TASK: Measure from pupil center to LENS BOTTOM EDGE (NOT frame rim)

KEY DISTINCTIONS:
- FRAME: The thick outer plastic/metal border you should IGNORE
- LENS: The actual glass/plastic lens inside the frame (this is your target)
- LENS EDGE: Where the lens material ends - usually 3-8mm ABOVE the frame bottom

INPUT DATA:
- Reference PD: ${pdValue}mm (verified)
- Bright blue markers show exact pupil centers

MEASUREMENT STEPS:

1. LOCATE PUPIL MARKERS:
   Find bright blue circular markers at pupil centers

2. CALCULATE SCALE:
   Scale = ${pdValue}mm ÷ (pixel distance between blue markers)

3. IDENTIFY LENS EDGE (CRITICAL):
   - Look INSIDE the frame boundary
   - Find where the transparent lens material ends
   - This appears as a subtle curved line or reflection change
   - Should be significantly HIGHER than the frame bottom
   - Typical lens edge is 60-80% down from pupil to frame bottom

4. MEASURE TO LENS EDGE:
   - Start: Center of bright blue pupil marker
   - End: Lens bottom edge (NOT frame bottom)

5. MEASURE ACCURATELY:
   - Be as precise as possible with your measurements
   - Use the exact same methodology for both eyes
   - Report your actual findings without adjusting to expected ranges

RETURN FORMAT:
{
  "leftOcularHeight": measurement_to_LENS_edge_in_mm,
  "rightOcularHeight": measurement_to_LENS_edge_in_mm,
  "confidence": 0_to_1_score,
  "analysisNotes": "Explicitly describe finding the lens edge vs frame boundary"
}`;

      const contents = [
        {
          inlineData: {
            data: imageBytes.toString("base64"),
            mimeType: "image/jpeg",
          },
        },
        prompt,
      ];

      const response = await ai.models.generateContent({
        model: "gemini-2.5-flash",
        config: {
          temperature: 0.0, // Maximum determinism
          topP: 0.1,        // Reduce randomness
          responseMimeType: "application/json",
          responseSchema: {
            type: "object",
            properties: {
              leftOcularHeight: { type: "number" },
              rightOcularHeight: { type: "number" },
              confidence: { type: "number" },
              analysisNotes: { type: "string" },
            },
            required: ["leftOcularHeight", "rightOcularHeight", "confidence", "analysisNotes"],
          },
        },
        contents: contents,
      });

      const rawJson = response.text;
      console.log(`Gemini ocular height analysis (attempt ${attempt}): ${rawJson}`);

      if (rawJson) {
        const data: OcularHeightResult = JSON.parse(rawJson);
        
        // Validate the result
        if (validateOcularHeight(data)) {
          console.log(`Ocular height analysis successful on attempt ${attempt}`);
          return data;
        } else {
          console.log(`Validation failed on attempt ${attempt}, retrying...`);
          lastError = new Error(`Invalid measurement ranges or inconsistent values on attempt ${attempt}`);
          continue;
        }
      } else {
        lastError = new Error("Empty response from Gemini model");
        continue;
      }
    } catch (error) {
      console.error(`Gemini analysis error on attempt ${attempt}:`, error);
      lastError = error as Error;
      
      // Wait before retry (exponential backoff)
      if (attempt < maxRetries) {
        await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
      }
    }
  }
  
  throw new Error(`Failed to analyze ocular height after ${maxRetries} attempts: ${lastError?.message}`);
}