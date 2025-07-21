# Pupillary Distance Measurement Application

## Overview

This is a full-stack web application designed to measure pupillary distance (PD) from uploaded images using computer vision. The application combines a React frontend with an Express.js backend, utilizing Python-based image processing with OpenCV Haar cascades and robust AprilTag pattern detection for accurate measurements.

## Recent Changes (July 21, 2025)

✓ **Replit Migration Complete** - Successfully migrated from Replit Agent to standard Replit environment
✓ **Python Packages Verified** - All computer vision dependencies (opencv-python, mediapipe, pupil-apriltags, numpy) working
✓ **Monocular PD Feature Added** - Enhanced PD measurement to include left and right monocular PD calculations
✓ **Nose Bridge Detection** - Implemented nose bridge center detection using MediaPipe facial landmarks
✓ **Database Schema Updated** - Added noseBridgeX, noseBridgeY, leftMonocularPd, rightMonocularPd fields
✓ **Complete API Integration** - Full stack now returns both total PD and individual monocular PD measurements
✓ **Visual Enhancements** - Processed images now show nose bridge marker and monocular PD lines

## Previous Changes (July 19, 2025)

✓ **Fixed file upload issues** - Corrected FormData handling in apiRequest function
✓ **Implemented robust computer vision** - Created multi-method eye detection using OpenCV Haar cascades, HoughCircles, and proportional fallback
✓ **Accurate overlay positioning** - Precise 1-pixel pupil markers and AprilTag bounding boxes
✓ **Successful test results** - Processed test image showing 76.0mm PD measurement with accurate visual overlays
✓ **Real AprilTag detection** - Implemented proper `pupil-apriltags` library with strict validation (no AprilTag = no analysis)
✓ **Enhanced precision** - AprilTag corners, ID detection, confidence scores, and memory-optimized processing
✓ **Advanced pupil detection** - Multi-method approach using darkness detection, HoughCircles, adaptive thresholding, circularity scoring, and gradient-based edge detection
✓ **Improved accuracy** - Refined PD measurements from initial estimates to precise 57.9mm with accurate pupil positioning

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Full-Stack TypeScript Application
- **Frontend**: React with TypeScript, Vite for bundling
- **Backend**: Express.js with TypeScript
- **Database**: PostgreSQL with Drizzle ORM
- **Image Processing**: Python scripts using MediaPipe and AprilTag detection
- **UI Framework**: shadcn/ui components with Radix UI primitives and Tailwind CSS

### Monorepo Structure
The application follows a monorepo pattern with shared TypeScript definitions:
- `client/` - React frontend application
- `server/` - Express.js backend API
- `shared/` - Shared TypeScript schemas and types
- Unified build process with Vite for frontend and esbuild for backend

## Key Components

### Database Layer (Drizzle ORM)
- **Schema Definition**: Located in `shared/schema.ts` with Zod validation
- **Tables**: 
  - `users` - User authentication and management
  - `measurements` - PD measurement results and metadata
- **Database Provider**: Configured for PostgreSQL with Neon serverless
- **Migrations**: Managed through Drizzle Kit in `./migrations` directory

### Backend API (Express.js)
- **File Upload Handling**: Multer middleware for image uploads (10MB limit, JPG/PNG/WEBP)
- **Image Processing Pipeline**: Spawns Python subprocess for computer vision processing
- **Storage Abstraction**: Interface-based storage with in-memory implementation (ready for database integration)
- **Development Middleware**: Request logging, error handling, and Vite integration

### Frontend Application (React)
- **State Management**: TanStack Query for server state management
- **Routing**: Wouter for lightweight client-side routing
- **UI Components**: Comprehensive shadcn/ui component library
- **File Upload**: Custom drag-and-drop file upload component with progress tracking
- **Form Handling**: React Hook Form with Zod validation

### Image Processing Engine (Python)
- **Computer Vision**: MediaPipe for facial landmark detection
- **AprilTag Detection**: For scale reference and measurement accuracy
- **Pupil Detection**: Extracts pupil coordinates for PD calculation
- **Error Handling**: Comprehensive validation and error reporting

### Manual Measurement Tools
- **Interactive Image Editor**: Canvas-based editor with zoom (20%-500%) and pan functionality
- **Precise Line Placement**: 1-pixel accuracy for frame bottom line positioning
- **Real-time Calculations**: Automatic ocular height computation based on pupil-to-line distance
- **Visual Feedback**: Pupil markers, measurement lines, and coordinate display

## Data Flow

### Measurement Process
1. **File Upload**: User selects and uploads an image through the React frontend
2. **Server Processing**: Express server receives the file via multer middleware
3. **Python Integration**: Server spawns Python script to process the image
4. **Computer Vision Analysis**: 
   - MediaPipe detects facial landmarks and pupil locations
   - AprilTag detector looks for scale reference markers
   - Calculate pixel distance between pupils
   - Apply scale factor if AprilTag is detected
5. **Result Storage**: Measurement data is saved to database with metadata
6. **Response**: Frontend receives processed results and displays to user

### Authentication Flow
- User registration and login system (schema defined, implementation pending)
- Session management with PostgreSQL session store
- Protected routes for measurement history

## External Dependencies

### Core Framework Dependencies
- **React**: Frontend framework with TypeScript support
- **Express.js**: Backend web server
- **Drizzle ORM**: Type-safe database operations
- **TanStack Query**: Server state management
- **Zod**: Runtime type validation

### UI and Styling
- **Tailwind CSS**: Utility-first CSS framework
- **Radix UI**: Accessible component primitives
- **shadcn/ui**: Pre-built component library
- **Lucide React**: Icon library

### Image Processing
- **Multer**: File upload handling
- **Python Dependencies**: MediaPipe, OpenCV, AprilTag detection libraries

### Development Tools
- **Vite**: Frontend build tool with HMR
- **esbuild**: Backend bundling
- **TypeScript**: Type safety across the stack
- **ESLint/Prettier**: Code quality and formatting

## Deployment Strategy

### Build Process
- **Frontend**: Vite builds React app to `dist/public`
- **Backend**: esbuild bundles Express server to `dist/index.js`
- **Database**: Drizzle migrations for schema deployment
- **Python Scripts**: Deployed alongside server code

### Environment Configuration
- **Database URL**: PostgreSQL connection string
- **File Storage**: Local uploads directory (configurable for cloud storage)
- **Python Environment**: Requires Python 3 with computer vision libraries

### Production Considerations
- Static file serving through Express
- File upload limits and validation
- Error handling and logging
- Database connection pooling
- Python subprocess management

### Development vs Production
- **Development**: Vite dev server with HMR, detailed logging
- **Production**: Optimized builds, compressed assets, production error handling
- **Replit Integration**: Special handling for Replit development environment