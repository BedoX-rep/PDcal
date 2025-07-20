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

      const prompt = `PRECISE OCULAR HEIGHT MEASUREMENT SYSTEM

IMAGE ANALYSIS GUIDE:
This is a processed eye measurement image with:
- Green horizontal lines marking the pupil locations (pupils are at the ENDS of these green lines)
- PD reference measurement: ${pdValue}mm
- Scale conversion: Use the scale information visible in the image (typically shown as "Scale: X.XXXmm/px")
- Yellow AprilTag for scale reference

IMPORTANT: 
- Analyze the image at its original resolution - do not resize or modify the image
- Use the exact pixel coordinates and measurements as they appear in the provided image
- Preserve all scale relationships based on the original image dimensions

MEASUREMENT PROTOCOL:

1. LOCATE PUPILS:
   - Find the END POINTS of the green horizontal lines
   - These endpoints mark the exact pupil centers
   - Left pupil: left end of left green line
   - Right pupil: right end of right green line

2. SCALE CALCULATION:
   - Look for scale text in image (e.g., "Scale: 0.429mm/px")
   - If scale not visible, calculate: ${pdValue}mm ÷ (pixel distance between pupil endpoints)

3. MEASURE OCULAR HEIGHT:
   - Start: Exact endpoint of each green line (pupil center)
   - End: Bottom edge of the LENS (not the frame) directly below each pupil
   - Measure vertically downward from pupil to the lens bottom edge
   - Count pixels precisely

4. CALCULATION:
   - Convert pixel measurements to millimeters using the scale factor
   - Use identical methodology for both eyes
   - Report exact measurements

CRITICAL REQUIREMENTS:
- Measure to the LENS bottom edge (NOT the frame - measure to where the lens material ends)
- Use green line endpoints as exact pupil locations
- Be precise with pixel counting
- Apply consistent scale conversion

OUTPUT FORMAT:
{
  "leftOcularHeight": exact_measurement_in_mm,
  "rightOcularHeight": exact_measurement_in_mm,
  "confidence": measurement_confidence_0_to_1,
  "analysisNotes": "Detailed pixel counts, scale used, and measurement methodology"
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