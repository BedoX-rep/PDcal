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

export async function analyzeOcularHeight(imagePath: string, pdValue: number): Promise<OcularHeightResult> {
  try {
    const imageBytes = fs.readFileSync(imagePath);

    const prompt = `You are an expert optometry assistant analyzing an eye measurement image. 

This image has already been processed for pupillary distance (PD) measurement, which is ${pdValue}mm.

Please analyze the ocular height (vertical distance from the pupil center to the bottom edge of the lens frame or the lower eyelid) for both eyes in this image.

IMPORTANT INSTRUCTIONS:
1. Look for the processed markers showing pupil locations (likely small circles or dots)
2. Measure the vertical distance from each pupil center to the bottom edge visible in the image
3. Use the known PD of ${pdValue}mm as a reference scale to calculate ocular heights
4. Provide measurements in millimeters for both left and right eyes
5. Be precise and explain your measurement approach

Respond with JSON in this exact format:
{
  "leftOcularHeight": number_in_mm,
  "rightOcularHeight": number_in_mm, 
  "confidence": confidence_score_0_to_1,
  "analysisNotes": "detailed_explanation_of_measurement_method"
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
      model: "gemini-2.5-pro",
      config: {
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
    console.log(`Gemini ocular height analysis: ${rawJson}`);

    if (rawJson) {
      const data: OcularHeightResult = JSON.parse(rawJson);
      return data;
    } else {
      throw new Error("Empty response from Gemini model");
    }
  } catch (error) {
    console.error("Gemini analysis error:", error);
    throw new Error(`Failed to analyze ocular height: ${error}`);
  }
}