import { useState, useRef, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { 
  Ruler, 
  Save, 
  RotateCcw, 
  ZoomIn, 
  ZoomOut, 
  Hand,
  Target,
  ArrowLeft
} from "lucide-react";

interface FrameSizeEditorProps {
  imageSrc: string;
  onSave: (leftLineX: number, rightLineX: number, frameWidth: number) => void;
  onCancel: () => void;
}

export function FrameSizeEditor({ imageSrc, onSave, onCancel }: FrameSizeEditorProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [lastPanPoint, setLastPanPoint] = useState({ x: 0, y: 0 });
  
  // Line states
  const [leftLine, setLeftLine] = useState<number | null>(null);
  const [rightLine, setRightLine] = useState<number | null>(null);
  const [frameWidth, setFrameWidth] = useState<string>("120");
  const [isDragging, setIsDragging] = useState<'left' | 'right' | null>(null);

  // Load image
  useEffect(() => {
    const img = new Image();
    img.onload = () => {
      setImage(img);
      // Center the image initially
      if (containerRef.current) {
        const containerRect = containerRef.current.getBoundingClientRect();
        setPan({
          x: (containerRect.width - img.width * zoom) / 2,
          y: (containerRect.height - img.height * zoom) / 2
        });
      }
    };
    img.src = imageSrc;
  }, [imageSrc, zoom]);

  // Draw canvas
  useEffect(() => {
    if (!image || !canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size to container size
    if (containerRef.current) {
      const containerRect = containerRef.current.getBoundingClientRect();
      canvas.width = containerRect.width;
      canvas.height = containerRect.height;
    }

    // Clear canvas
    ctx.fillStyle = '#f8fafc';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw image
    ctx.drawImage(
      image,
      pan.x,
      pan.y,
      image.width * zoom,
      image.height * zoom
    );

    // Draw vertical lines
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);

    if (leftLine !== null) {
      const lineX = pan.x + leftLine * zoom;
      ctx.beginPath();
      ctx.moveTo(lineX, 0);
      ctx.lineTo(lineX, canvas.height);
      ctx.stroke();
      
      // Draw label
      ctx.fillStyle = '#3b82f6';
      ctx.font = '14px sans-serif';
      ctx.fillText('Left Edge', lineX + 5, 20);
    }

    if (rightLine !== null) {
      const lineX = pan.x + rightLine * zoom;
      ctx.beginPath();
      ctx.moveTo(lineX, 0);
      ctx.lineTo(lineX, canvas.height);
      ctx.stroke();
      
      // Draw label
      ctx.fillStyle = '#3b82f6';
      ctx.font = '14px sans-serif';
      ctx.fillText('Right Edge', lineX + 5, 40);
    }

    // Draw measurement line between lines
    if (leftLine !== null && rightLine !== null) {
      const leftX = pan.x + leftLine * zoom;
      const rightX = pan.x + rightLine * zoom;
      const centerY = 60;
      
      ctx.setLineDash([]);
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 3;
      
      // Horizontal measurement line
      ctx.beginPath();
      ctx.moveTo(leftX, centerY);
      ctx.lineTo(rightX, centerY);
      ctx.stroke();
      
      // End markers
      ctx.beginPath();
      ctx.moveTo(leftX, centerY - 10);
      ctx.lineTo(leftX, centerY + 10);
      ctx.moveTo(rightX, centerY - 10);
      ctx.lineTo(rightX, centerY + 10);
      ctx.stroke();
      
      // Distance label
      const pixelDistance = Math.abs(rightLine - leftLine);
      ctx.fillStyle = '#ef4444';
      ctx.font = 'bold 16px sans-serif';
      const text = `${pixelDistance.toFixed(1)} pixels`;
      const textWidth = ctx.measureText(text).width;
      ctx.fillText(text, (leftX + rightX - textWidth) / 2, centerY - 15);
    }
  }, [image, zoom, pan, leftLine, rightLine]);

  const getCanvasCoordinates = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    // Convert to image coordinates
    const imageX = (x - pan.x) / zoom;
    const imageY = (y - pan.y) / zoom;
    
    return { x: imageX, y: imageY };
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (e.shiftKey) {
      // Pan mode
      setIsPanning(true);
      setLastPanPoint({ x: e.clientX, y: e.clientY });
      return;
    }

    const coords = getCanvasCoordinates(e);
    
    // Check if clicking near existing lines for dragging
    if (leftLine !== null && Math.abs(coords.x - leftLine) < 10) {
      setIsDragging('left');
      return;
    }
    if (rightLine !== null && Math.abs(coords.x - rightLine) < 10) {
      setIsDragging('right');
      return;
    }

    // Place new line
    if (leftLine === null) {
      setLeftLine(coords.x);
    } else if (rightLine === null) {
      setRightLine(coords.x);
    } else {
      // Replace closest line
      const leftDist = Math.abs(coords.x - leftLine);
      const rightDist = Math.abs(coords.x - rightLine);
      if (leftDist < rightDist) {
        setLeftLine(coords.x);
      } else {
        setRightLine(coords.x);
      }
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isPanning) {
      const deltaX = e.clientX - lastPanPoint.x;
      const deltaY = e.clientY - lastPanPoint.y;
      setPan(prev => ({
        x: prev.x + deltaX,
        y: prev.y + deltaY
      }));
      setLastPanPoint({ x: e.clientX, y: e.clientY });
      return;
    }

    if (isDragging) {
      const coords = getCanvasCoordinates(e);
      if (isDragging === 'left') {
        setLeftLine(coords.x);
      } else {
        setRightLine(coords.x);
      }
    }
  };

  const handleMouseUp = () => {
    setIsPanning(false);
    setIsDragging(null);
  };

  const handleZoom = (direction: 'in' | 'out') => {
    const newZoom = direction === 'in' ? zoom * 1.2 : zoom / 1.2;
    setZoom(Math.max(0.1, Math.min(5, newZoom)));
  };

  const handleReset = () => {
    setLeftLine(null);
    setRightLine(null);
    setFrameWidth("120");
  };

  const handleSave = () => {
    if (leftLine === null || rightLine === null || !frameWidth) {
      return;
    }
    
    const frameWidthNum = parseFloat(frameWidth);
    if (isNaN(frameWidthNum) || frameWidthNum <= 0) {
      return;
    }

    onSave(leftLine, rightLine, frameWidthNum);
  };

  const canSave = leftLine !== null && rightLine !== null && frameWidth && parseFloat(frameWidth) > 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Ruler className="h-5 w-5" />
            Frame Size Analysis
          </CardTitle>
          <CardDescription>
            Draw vertical lines on the left and right edges of the frame, then input the frame width
          </CardDescription>
        </CardHeader>
      </Card>

      {/* Controls */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex flex-wrap items-center gap-4">
            <div className="flex items-center gap-2">
              <Button variant="outline" size="sm" onClick={() => handleZoom('out')}>
                <ZoomOut className="h-4 w-4" />
              </Button>
              <span className="text-sm font-medium w-16 text-center">
                {Math.round(zoom * 100)}%
              </span>
              <Button variant="outline" size="sm" onClick={() => handleZoom('in')}>
                <ZoomIn className="h-4 w-4" />
              </Button>
            </div>
            
            <div className="flex items-center gap-2">
              <Hand className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm text-muted-foreground">Hold Shift + drag to pan</span>
            </div>
            
            <div className="flex items-center gap-2">
              <Label htmlFor="frameWidth">Frame Width (mm):</Label>
              <Input
                id="frameWidth"
                type="number"
                value={frameWidth}
                onChange={(e) => setFrameWidth(e.target.value)}
                className="w-24"
                min="1"
                step="0.1"
              />
            </div>
            
            <Button variant="outline" size="sm" onClick={handleReset}>
              <RotateCcw className="h-4 w-4 mr-2" />
              Reset Lines
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Status */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex flex-wrap items-center gap-4">
            <div className="flex items-center gap-2">
              <Badge variant={leftLine !== null ? "default" : "outline"}>
                Left Edge: {leftLine !== null ? "Set" : "Not set"}
              </Badge>
              <Badge variant={rightLine !== null ? "default" : "outline"}>
                Right Edge: {rightLine !== null ? "Set" : "Not set"}
              </Badge>
            </div>
            
            {leftLine !== null && rightLine !== null && (
              <div className="text-sm text-muted-foreground">
                Pixel distance: {Math.abs(rightLine - leftLine).toFixed(1)} pixels
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Canvas */}
      <Card>
        <CardContent className="pt-6">
          <div 
            ref={containerRef}
            className="relative w-full h-96 border rounded-lg overflow-hidden bg-slate-50"
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
          
          <div className="mt-4 text-sm text-muted-foreground space-y-1">
            <p>• Click to place vertical lines at the left and right frame edges</p>
            <p>• Click and drag existing lines to adjust their position</p>
            <p>• Hold Shift and drag to pan the image</p>
            <p>• Use zoom controls to get precise line placement</p>
          </div>
        </CardContent>
      </Card>

      {/* Action Buttons */}
      <div className="flex justify-between">
        <Button variant="outline" onClick={onCancel}>
          <ArrowLeft className="h-4 w-4 mr-2" />
          Cancel
        </Button>
        
        <Button 
          onClick={handleSave}
          disabled={!canSave}
          size="lg"
        >
          <Save className="h-4 w-4 mr-2" />
          Analyze with Frame Size
        </Button>
      </div>
    </div>
  );
}