import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Button } from './button';
import { Card, CardContent, CardHeader, CardTitle } from './card';
import { ZoomIn, ZoomOut, RotateCcw, Save } from 'lucide-react';

interface ImageEditorProps {
  imageSrc: string;
  onSave: (leftLineY: number, rightLineY: number, zoomLevel: number, imageWidth: number, imageHeight: number) => void;
  onCancel: () => void;
  leftPupil: { x: number; y: number };
  rightPupil: { x: number; y: number };
}

export function ImageEditor({ imageSrc, onSave, onCancel, leftPupil, rightPupil }: ImageEditorProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [leftLineY, setLeftLineY] = useState<number | null>(null);
  const [rightLineY, setRightLineY] = useState<number | null>(null);
  const [isDragging, setIsDragging] = useState<'left' | 'right' | null>(null);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0 });
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [lastPanPoint, setLastPanPoint] = useState({ x: 0, y: 0 });

  const imageRef = useRef<HTMLImageElement>(new Image());

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const img = imageRef.current;
    img.onload = () => {
      setImageDimensions({ width: img.width, height: img.height });
      setImageLoaded(true);
      
      // Set initial line positions to 80% down the image (common frame bottom position)
      const initialY = Math.floor(img.height * 0.8);
      setLeftLineY(initialY);
      setRightLineY(initialY);
      
      drawCanvas();
    };
    img.src = imageSrc;
  }, [imageSrc]);

  useEffect(() => {
    if (imageLoaded) {
      drawCanvas();
    }
  }, [zoomLevel, leftLineY, rightLineY, panOffset, imageLoaded]);

  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imageLoaded) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = imageRef.current;
    
    // Set canvas size
    canvas.width = Math.min(800, imageDimensions.width * zoomLevel);
    canvas.height = Math.min(600, imageDimensions.height * zoomLevel);

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Calculate image position with pan offset
    const imgWidth = imageDimensions.width * zoomLevel;
    const imgHeight = imageDimensions.height * zoomLevel;
    const imgX = (canvas.width - imgWidth) / 2 + panOffset.x;
    const imgY = (canvas.height - imgHeight) / 2 + panOffset.y;

    // Draw image
    ctx.drawImage(img, imgX, imgY, imgWidth, imgHeight);

    // Draw pupil markers
    const leftX = imgX + (leftPupil.x * zoomLevel);
    const leftY = imgY + (leftPupil.y * zoomLevel);
    const rightX = imgX + (rightPupil.x * zoomLevel);
    const rightY = imgY + (rightPupil.y * zoomLevel);

    // Draw pupil circles
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(leftX, leftY, 5, 0, 2 * Math.PI);
    ctx.stroke();
    
    ctx.beginPath();
    ctx.arc(rightX, rightY, 5, 0, 2 * Math.PI);
    ctx.stroke();

    // Draw left eye frame line (red)
    if (leftLineY !== null) {
      const scaledLeftLineY = imgY + (leftLineY * zoomLevel);
      ctx.strokeStyle = '#ff0000';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(imgX, scaledLeftLineY);
      ctx.lineTo(leftX + 50, scaledLeftLineY); // Extend line to left of left pupil
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw vertical line from left pupil to left frame line
      ctx.strokeStyle = '#ffff00';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(leftX, leftY);
      ctx.lineTo(leftX, scaledLeftLineY);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Draw right eye frame line (blue)
    if (rightLineY !== null) {
      const scaledRightLineY = imgY + (rightLineY * zoomLevel);
      ctx.strokeStyle = '#0000ff';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(rightX - 50, scaledRightLineY); // Start line to right of right pupil
      ctx.lineTo(imgX + imgWidth, scaledRightLineY);
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw vertical line from right pupil to right frame line
      ctx.strokeStyle = '#ffff00';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(rightX, rightY);
      ctx.lineTo(rightX, scaledRightLineY);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [imageLoaded, zoomLevel, leftLineY, rightLineY, panOffset, imageDimensions, leftPupil, rightPupil]);

  const getCanvasCoordinates = (clientX: number, clientY: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };

    const rect = canvas.getBoundingClientRect();
    const canvasX = clientX - rect.left;
    const canvasY = clientY - rect.top;

    // Convert to image coordinates
    const imgWidth = imageDimensions.width * zoomLevel;
    const imgHeight = imageDimensions.height * zoomLevel;
    const imgX = (canvas.width - imgWidth) / 2 + panOffset.x;
    const imgY = (canvas.height - imgHeight) / 2 + panOffset.y;

    const imageX = (canvasX - imgX) / zoomLevel;
    const imageY = (canvasY - imgY) / zoomLevel;

    return { x: imageX, y: imageY };
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    const coords = getCanvasCoordinates(e.clientX, e.clientY);
    
    if (e.shiftKey) {
      // Pan mode
      setIsPanning(true);
      setLastPanPoint({ x: e.clientX, y: e.clientY });
    } else {
      // Line placement mode - determine which eye is closer
      if (coords.x >= 0 && coords.x <= imageDimensions.width && 
          coords.y >= 0 && coords.y <= imageDimensions.height) {
        
        // Determine which line to adjust based on click position
        const leftPupilDistance = Math.abs(coords.x - leftPupil.x);
        const rightPupilDistance = Math.abs(coords.x - rightPupil.x);
        
        if (leftPupilDistance < rightPupilDistance) {
          // Closer to left pupil, adjust left line
          setLeftLineY(coords.y);
          setIsDragging('left');
        } else {
          // Closer to right pupil, adjust right line
          setRightLineY(coords.y);
          setIsDragging('right');
        }
      }
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isPanning) {
      const deltaX = e.clientX - lastPanPoint.x;
      const deltaY = e.clientY - lastPanPoint.y;
      setPanOffset(prev => ({ x: prev.x + deltaX, y: prev.y + deltaY }));
      setLastPanPoint({ x: e.clientX, y: e.clientY });
    } else if (isDragging) {
      const coords = getCanvasCoordinates(e.clientX, e.clientY);
      if (coords.y >= 0 && coords.y <= imageDimensions.height) {
        if (isDragging === 'left') {
          setLeftLineY(coords.y);
        } else if (isDragging === 'right') {
          setRightLineY(coords.y);
        }
      }
    }
  };

  const handleMouseUp = () => {
    setIsDragging(null);
    setIsPanning(false);
  };

  const handleZoomIn = () => {
    setZoomLevel(prev => Math.min(prev * 1.5, 5));
  };

  const handleZoomOut = () => {
    setZoomLevel(prev => Math.max(prev / 1.5, 0.2));
  };

  const handleReset = () => {
    setZoomLevel(1);
    setPanOffset({ x: 0, y: 0 });
    const initialY = Math.floor(imageDimensions.height * 0.8);
    setLeftLineY(initialY);
    setRightLineY(initialY);
  };

  const handleSave = () => {
    if (leftLineY !== null && rightLineY !== null) {
      onSave(leftLineY, rightLineY, zoomLevel, imageDimensions.width, imageDimensions.height);
    }
  };

  return (
    <Card className="w-full max-w-4xl mx-auto">
      <CardHeader>
        <CardTitle>Manual Ocular Height Calculator</CardTitle>
        <div className="text-sm text-muted-foreground">
          Click and drag to position frame lines at the bottom of each lens. Red line for left eye, blue line for right eye.
          Hold Shift + drag to pan. Green circles show pupil positions.
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex gap-2 items-center justify-between">
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={handleZoomIn}>
              <ZoomIn className="w-4 h-4" />
              Zoom In
            </Button>
            <Button variant="outline" size="sm" onClick={handleZoomOut}>
              <ZoomOut className="w-4 h-4" />
              Zoom Out
            </Button>
            <Button variant="outline" size="sm" onClick={handleReset}>
              <RotateCcw className="w-4 h-4" />
              Reset
            </Button>
          </div>
          <div className="text-sm text-muted-foreground">
            Zoom: {(zoomLevel * 100).toFixed(0)}%
          </div>
        </div>

        <div 
          ref={containerRef}
          className="border rounded-lg overflow-hidden bg-gray-50 dark:bg-gray-900"
          style={{ maxHeight: '600px' }}
        >
          <canvas
            ref={canvasRef}
            className="cursor-crosshair"
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
          />
        </div>

        {(leftLineY !== null || rightLineY !== null) && (
          <div className="text-sm text-muted-foreground space-y-1">
            {leftLineY !== null && (
              <div>Left eye (red line) positioned at Y: {leftLineY.toFixed(1)}px</div>
            )}
            {rightLineY !== null && (
              <div>Right eye (blue line) positioned at Y: {rightLineY.toFixed(1)}px</div>
            )}
          </div>
        )}

        <div className="flex gap-2 justify-end">
          <Button variant="outline" onClick={onCancel}>
            Cancel
          </Button>
          <Button onClick={handleSave} disabled={leftLineY === null || rightLineY === null}>
            <Save className="w-4 h-4 mr-2" />
            Calculate Ocular Heights
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}