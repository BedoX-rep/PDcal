import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Button } from './button';
import { Card, CardContent, CardHeader, CardTitle } from './card';
import { ZoomIn, ZoomOut, RotateCcw, Save } from 'lucide-react';

interface ImageEditorProps {
  imageSrc: string;
  onSave: (lineY: number, zoomLevel: number, imageWidth: number, imageHeight: number) => void;
  onCancel: () => void;
  leftPupil: { x: number; y: number };
  rightPupil: { x: number; y: number };
}

export function ImageEditor({ imageSrc, onSave, onCancel, leftPupil, rightPupil }: ImageEditorProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [lineY, setLineY] = useState<number | null>(null);
  const [isDragging, setIsDragging] = useState(false);
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
      
      // Set initial line position to 80% down the image (common frame bottom position)
      setLineY(Math.floor(img.height * 0.8));
      
      drawCanvas();
    };
    img.src = imageSrc;
  }, [imageSrc]);

  useEffect(() => {
    if (imageLoaded) {
      drawCanvas();
    }
  }, [zoomLevel, lineY, panOffset, imageLoaded]);

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

    // Draw horizontal line if set
    if (lineY !== null) {
      const scaledLineY = imgY + (lineY * zoomLevel);
      ctx.strokeStyle = '#ff0000';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(imgX, scaledLineY);
      ctx.lineTo(imgX + imgWidth, scaledLineY);
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw vertical lines from pupils to frame line
      ctx.strokeStyle = '#ffff00';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      
      // Left eye vertical line
      ctx.beginPath();
      ctx.moveTo(leftX, leftY);
      ctx.lineTo(leftX, scaledLineY);
      ctx.stroke();
      
      // Right eye vertical line
      ctx.beginPath();
      ctx.moveTo(rightX, rightY);
      ctx.lineTo(rightX, scaledLineY);
      ctx.stroke();
      
      ctx.setLineDash([]);
    }
  }, [imageLoaded, zoomLevel, lineY, panOffset, imageDimensions, leftPupil, rightPupil]);

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
      // Line placement mode
      if (coords.x >= 0 && coords.x <= imageDimensions.width && 
          coords.y >= 0 && coords.y <= imageDimensions.height) {
        setLineY(coords.y);
        setIsDragging(true);
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
        setLineY(coords.y);
      }
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
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
    setLineY(Math.floor(imageDimensions.height * 0.8));
  };

  const handleSave = () => {
    if (lineY !== null) {
      onSave(lineY, zoomLevel, imageDimensions.width, imageDimensions.height);
    }
  };

  return (
    <Card className="w-full max-w-4xl mx-auto">
      <CardHeader>
        <CardTitle>Manual Ocular Height Calculator</CardTitle>
        <div className="text-sm text-muted-foreground">
          Click and drag to position the red line at the bottom of the frame. 
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

        {lineY !== null && (
          <div className="text-sm text-muted-foreground">
            Frame bottom line positioned at Y: {lineY.toFixed(1)}px
          </div>
        )}

        <div className="flex gap-2 justify-end">
          <Button variant="outline" onClick={onCancel}>
            Cancel
          </Button>
          <Button onClick={handleSave} disabled={lineY === null}>
            <Save className="w-4 h-4 mr-2" />
            Calculate Ocular Height
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}