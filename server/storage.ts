import { users, measurements, type User, type InsertUser, type Measurement, type InsertMeasurement } from "@shared/schema";

export interface IStorage {
  getUser(id: number): Promise<User | undefined>;
  getUserByUsername(username: string): Promise<User | undefined>;
  createUser(user: InsertUser): Promise<User>;
  createMeasurement(measurement: InsertMeasurement): Promise<Measurement>;
  getMeasurement(id: number): Promise<Measurement | undefined>;
  getAllMeasurements(): Promise<Measurement[]>;
  updateMeasurement(id: number, updates: Partial<Measurement>): Promise<Measurement>;
}

export class MemStorage implements IStorage {
  private users: Map<number, User>;
  private measurements: Map<number, Measurement>;
  private currentUserId: number;
  private currentMeasurementId: number;

  constructor() {
    this.users = new Map();
    this.measurements = new Map();
    this.currentUserId = 1;
    this.currentMeasurementId = 1;
  }

  async getUser(id: number): Promise<User | undefined> {
    return this.users.get(id);
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    return Array.from(this.users.values()).find(
      (user) => user.username === username,
    );
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const id = this.currentUserId++;
    const user: User = { ...insertUser, id };
    this.users.set(id, user);
    return user;
  }

  async createMeasurement(insertMeasurement: InsertMeasurement): Promise<Measurement> {
    const id = this.currentMeasurementId++;
    const measurement: Measurement = { 
      id,
      originalFilename: insertMeasurement.originalFilename,
      processedImagePath: insertMeasurement.processedImagePath ?? null,
      pdValue: insertMeasurement.pdValue ?? null,
      leftPupilX: insertMeasurement.leftPupilX ?? null,
      leftPupilY: insertMeasurement.leftPupilY ?? null,
      rightPupilX: insertMeasurement.rightPupilX ?? null,
      rightPupilY: insertMeasurement.rightPupilY ?? null,
      noseBridgeX: insertMeasurement.noseBridgeX ?? null,
      noseBridgeY: insertMeasurement.noseBridgeY ?? null,
      leftMonocularPd: insertMeasurement.leftMonocularPd ?? null,
      rightMonocularPd: insertMeasurement.rightMonocularPd ?? null,
      pixelDistance: insertMeasurement.pixelDistance ?? null,
      scaleFactor: insertMeasurement.scaleFactor ?? null,
      apriltagDetected: insertMeasurement.apriltagDetected ?? null,
      pupilsDetected: insertMeasurement.pupilsDetected ?? null,
      leftOcularHeight: insertMeasurement.leftOcularHeight ?? null,
      rightOcularHeight: insertMeasurement.rightOcularHeight ?? null,
      ocularHeightAnalyzed: insertMeasurement.ocularHeightAnalyzed ?? null,
      errorMessage: insertMeasurement.errorMessage ?? null,
      createdAt: new Date()
    };
    this.measurements.set(id, measurement);
    return measurement;
  }

  async getMeasurement(id: number): Promise<Measurement | undefined> {
    return this.measurements.get(id);
  }

  async getAllMeasurements(): Promise<Measurement[]> {
    return Array.from(this.measurements.values());
  }

  async updateMeasurement(id: number, updates: Partial<Measurement>): Promise<Measurement> {
    const existing = this.measurements.get(id);
    if (!existing) {
      throw new Error(`Measurement with id ${id} not found`);
    }
    
    const updated = { ...existing, ...updates };
    this.measurements.set(id, updated);
    return updated;
  }
}

export const storage = new MemStorage();
