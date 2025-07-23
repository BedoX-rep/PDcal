import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { FileUpload } from "@/components/ui/file-upload";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/hooks/use-auth";
import { apiRequest } from "@/lib/queryClient";
import MeasurementResults from "./measurement-results";
import { MeasurementHistory } from "./measurement-history";
import { FrameSizeEditor } from "@/components/ui/frame-size-editor";
import { 
  Eye, 
  Info, 
  Shield, 
  CheckCircle, 
  AlertTriangle, 
  Ruler, 
  RotateCcw,
  Save,
  FileText,
  Edit,
  LogOut,
  User,
  Upload,
  History,
  Target,
  Zap
} from "lucide-react";
import type { Measurement } from "@shared/schema";
interface ProcessingResult {
  success: boolean;
  measurement: Measurement;
  result: {
    pd_value: number;
    left_pupil: { x: number; y: number };
    right_pupil: { x: number; y: number };
    pixel_distance: number;
    scale_factor: number;
    processed_image_path: string;
  };
}

export default function EnhancedHome() {
  const [activeTab, setActiveTab] = useState("upload");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [processingResult, setProcessingResult] = useState<ProcessingResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showAnalysisSelection, setShowAnalysisSelection] = useState(false);
  const [analysisMode, setAnalysisMode] = useState<'apriltag' | 'framesize' | null>(null);
  const [showFrameSizeEditor, setShowFrameSizeEditor] = useState(false);
  const { toast } = useToast();
  const { user, signOut } = useAuth();

  const uploadMutation = useMutation({
    mutationFn: async ({ file, mode }: { file: File; mode: 'apriltag' | 'framesize' }) => {
      const formData = new FormData();
      formData.append('image', file);
      formData.append('analysisMode', mode);

      const response = await apiRequest("POST", "/api/measurements", formData);
      return response.json();
    },
    onSuccess: (data) => {
      setProcessingResult(data);
      setActiveTab("results");
      setShowAnalysisSelection(false);
      setAnalysisMode(null);
      toast({
        title: "Success!",
        description: "Image processed successfully. PD measured: " + data.result.pd_value + "mm",
      });
    },
    onError: (error) => {
      setError(error.message);
      setShowAnalysisSelection(false);
      toast({
        title: "Processing Failed",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setError(null);
    setShowAnalysisSelection(true); // Show analysis mode selection after file upload
  };

  const handleAnalysisModeSelect = (mode: 'apriltag' | 'framesize') => {
    setAnalysisMode(mode);
    if (mode === 'apriltag') {
      // Direct processing for AprilTag mode
      if (selectedFile) {
        uploadMutation.mutate({ file: selectedFile, mode });
      }
    } else {
      // Frame size mode - show the frame size editor
      setShowAnalysisSelection(false);
      setShowFrameSizeEditor(true);
    }
  };

  const frameSizeAnalysisMutation = useMutation({
    mutationFn: async ({ file, leftLineX, rightLineX, frameWidth }: { 
      file: File; 
      leftLineX: number; 
      rightLineX: number; 
      frameWidth: number; 
    }) => {
      const formData = new FormData();
      formData.append('image', file);
      formData.append('analysisMode', 'framesize');
      formData.append('leftLineX', leftLineX.toString());
      formData.append('rightLineX', rightLineX.toString());
      formData.append('frameWidth', frameWidth.toString());

      const response = await apiRequest("POST", "/api/measurements", formData);
      return response.json();
    },
    onSuccess: (data) => {
      setProcessingResult(data);
      setActiveTab("results");
      setShowFrameSizeEditor(false);
      setAnalysisMode(null);
      toast({
        title: "Success!",
        description: "Frame size analysis complete. PD measured: " + data.result.pd_value + "mm",
      });
    },
    onError: (error) => {
      setError(error.message);
      toast({
        title: "Frame Size Analysis Failed",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const handleFrameSizeAnalysis = (leftLineX: number, rightLineX: number, frameWidth: number) => {
    if (!selectedFile) return;
    frameSizeAnalysisMutation.mutate({ 
      file: selectedFile, 
      leftLineX, 
      rightLineX, 
      frameWidth 
    });
  };

  const handleFrameSizeCancel = () => {
    setShowFrameSizeEditor(false);
    setShowAnalysisSelection(true);
  };

  const handleStartOver = () => {
    setSelectedFile(null);
    setProcessingResult(null);
    setError(null);
    setShowAnalysisSelection(false);
    setAnalysisMode(null);
    setShowFrameSizeEditor(false);
    setActiveTab("upload");
  };

  const handleSaveComplete = () => {
    // Refresh history and stay on results tab
    toast({
      title: "Saved!",
      description: "Measurement has been saved to your history.",
    });
  };

  if (!user) {
    return (
      <div className="container mx-auto p-6">
        <div className="text-center">Loading...</div>
      </div>
    );
  }

  // Show frame size editor if mode is selected
  if (showFrameSizeEditor && selectedFile) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900">
        {/* Header */}
        <div className="bg-white dark:bg-gray-800 shadow-sm border-b">
          <div className="container mx-auto px-6 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg">
                  <Eye className="h-6 w-6 text-white" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                    PD Measurement Tool
                  </h1>
                  <p className="text-sm text-muted-foreground">Frame Size Analysis Mode</p>
                </div>
              </div>
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2 text-sm">
                  <User className="h-4 w-4" />
                  <span className="font-medium">{user.email}</span>
                </div>
                <Button variant="outline" size="sm" onClick={signOut}>
                  <LogOut className="h-4 w-4 mr-2" />
                  Sign Out
                </Button>
              </div>
            </div>
          </div>
        </div>

        {/* Frame Size Editor */}
        <div className="container mx-auto p-6">
          <FrameSizeEditor
            imageSrc={URL.createObjectURL(selectedFile)}
            onSave={handleFrameSizeAnalysis}
            onCancel={handleFrameSizeCancel}
          />
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 shadow-sm border-b">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg">
                <Eye className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  PD Measurement Tool
                </h1>
                <p className="text-sm text-muted-foreground">Professional pupillary distance analysis</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2 text-sm">
                <User className="h-4 w-4" />
                <span className="font-medium">{user.email}</span>
              </div>
              <Button variant="outline" size="sm" onClick={signOut}>
                <LogOut className="h-4 w-4 mr-2" />
                Sign Out
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="container mx-auto p-6">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="upload" className="flex items-center gap-2">
              <Upload className="h-4 w-4" />
              Upload & Process
            </TabsTrigger>
            <TabsTrigger value="results" className="flex items-center gap-2" disabled={!processingResult}>
              <Target className="h-4 w-4" />
              Measurement Results
            </TabsTrigger>
            <TabsTrigger value="history" className="flex items-center gap-2">
              <History className="h-4 w-4" />
              Measurement History
            </TabsTrigger>
          </TabsList>

          {/* Upload Tab */}
          <TabsContent value="upload" className="space-y-6">
            <Card>
              <CardHeader className="text-center space-y-2">
                <CardTitle className="flex items-center justify-center gap-2 text-2xl">
                  <Upload className="h-6 w-6" />
                  Upload Image for PD Measurement
                </CardTitle>
                <CardDescription className="text-base">
                  Upload a clear photo for PD measurement with your choice of analysis method
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Upload Area */}
                <FileUpload onFileSelect={handleFileSelect} />

                {/* Selected File Display */}
                {selectedFile && !showAnalysisSelection && (
                  <Card className="border-green-200 bg-green-50/50">
                    <CardContent className="pt-6">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          <div className="p-2 bg-green-100 rounded-lg">
                            <FileText className="h-5 w-5 text-green-600" />
                          </div>
                          <div>
                            <p className="font-medium">{selectedFile.name}</p>
                            <p className="text-sm text-muted-foreground">
                              {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                            </p>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                )}

                {/* Analysis Mode Selection */}
                {selectedFile && showAnalysisSelection && (
                  <Card className="border-blue-200 bg-blue-50/50">
                    <CardHeader className="text-center">
                      <CardTitle className="flex items-center justify-center gap-2">
                        <Target className="h-5 w-5" />
                        Choose Analysis Method
                      </CardTitle>
                      <CardDescription>
                        Select how you want to measure the pupillary distance
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {/* AprilTag Analysis */}
                        <Card className="cursor-pointer hover:shadow-md transition-shadow border-2 border-transparent hover:border-blue-300"
                              onClick={() => handleAnalysisModeSelect('apriltag')}>
                          <CardContent className="p-6 text-center space-y-3">
                            <div className="p-3 bg-blue-100 rounded-lg mx-auto w-fit">
                              <Zap className="h-8 w-8 text-blue-600" />
                            </div>
                            <h3 className="font-semibold text-lg">AprilTag Analysis</h3>
                            <p className="text-sm text-muted-foreground">
                              Automatic scale detection using AprilTag reference markers in your image
                            </p>
                            <Badge variant="secondary" className="bg-green-100 text-green-800">
                              <CheckCircle className="h-3 w-3 mr-1" />
                              Fully Automatic
                            </Badge>
                          </CardContent>
                        </Card>

                        {/* Frame Size Analysis */}
                        <Card className="cursor-pointer hover:shadow-md transition-shadow border-2 border-transparent hover:border-purple-300"
                              onClick={() => handleAnalysisModeSelect('framesize')}>
                          <CardContent className="p-6 text-center space-y-3">
                            <div className="p-3 bg-purple-100 rounded-lg mx-auto w-fit">
                              <Ruler className="h-8 w-8 text-purple-600" />
                            </div>
                            <h3 className="font-semibold text-lg">Frame Size Analysis</h3>
                            <p className="text-sm text-muted-foreground">
                              Draw vertical lines on frame edges and input frame width for scale
                            </p>
                            <Badge variant="secondary" className="bg-blue-100 text-blue-800">
                              <Edit className="h-3 w-3 mr-1" />
                              Manual Setup
                            </Badge>
                          </CardContent>
                        </Card>
                      </div>
                      
                      <div className="flex justify-center">
                        <Button 
                          variant="outline" 
                          onClick={() => setShowAnalysisSelection(false)}
                          className="mt-4"
                        >
                          <RotateCcw className="h-4 w-4 mr-2" />
                          Choose Different Image
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                )}

                {/* Processing Progress */}
                {uploadMutation.isPending && (
                  <Card className="border-blue-200 bg-blue-50/50">
                    <CardContent className="pt-6">
                      <div className="space-y-4">
                        <div className="flex items-center space-x-2">
                          <Zap className="h-5 w-5 text-blue-600 animate-pulse" />
                          <span className="font-medium">Processing your image...</span>
                        </div>
                        <Progress value={75} className="h-2" />
                        <div className="text-sm text-muted-foreground space-y-1">
                          <p>• Detecting facial landmarks and pupils</p>
                          <p>• Looking for AprilTag reference markers</p>
                          <p>• Calculating accurate measurements</p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                )}

                {/* Error Display */}
                {error && (
                  <Card className="border-red-200 bg-red-50/50">
                    <CardContent className="pt-6">
                      <div className="flex items-start space-x-3">
                        <AlertTriangle className="h-5 w-5 text-red-600 mt-0.5" />
                        <div>
                          <p className="font-medium text-red-800">Processing Failed</p>
                          <p className="text-red-700 text-sm">{error}</p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                )}

                {/* Instructions */}
                <Card className="border-blue-200 bg-blue-50/50">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-lg">
                      <Info className="h-5 w-5" />
                      Instructions for Best Results
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <h4 className="font-medium flex items-center gap-2">
                          <CheckCircle className="h-4 w-4 text-green-600" />
                          Do:
                        </h4>
                        <ul className="text-sm space-y-1 pl-6">
                          <li>• Include an AprilTag reference in the image</li>
                          <li>• Ensure clear visibility of both eyes</li>
                          <li>• Use good lighting conditions</li>
                          <li>• Keep the face straight and centered</li>
                        </ul>
                      </div>
                      <div className="space-y-2">
                        <h4 className="font-medium flex items-center gap-2">
                          <AlertTriangle className="h-4 w-4 text-amber-600" />
                          Avoid:
                        </h4>
                        <ul className="text-sm space-y-1 pl-6">
                          <li>• Blurry or low-quality images</li>
                          <li>• Closed or partially visible eyes</li>
                          <li>• Extreme head tilting or rotation</li>
                          <li>• Poor lighting or shadows</li>
                        </ul>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Results Tab */}
          <TabsContent value="results" className="space-y-6">
            {processingResult ? (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-2xl font-bold">Your Measurement Results</h2>
                    <p className="text-muted-foreground">Review and save your PD measurement</p>
                  </div>
                  <Button variant="outline" onClick={handleStartOver}>
                    <RotateCcw className="h-4 w-4 mr-2" />
                    Start Over
                  </Button>
                </div>

                <MeasurementResults 
                  measurementData={processingResult}
                  onSave={handleSaveComplete}
                />
              </div>
            ) : (
              <Card>
                <CardContent className="pt-6 text-center">
                  <div className="space-y-4">
                    <Target className="h-16 w-16 text-muted-foreground mx-auto" />
                    <div>
                      <h3 className="text-lg font-medium">No measurement data</h3>
                      <p className="text-muted-foreground">
                        Process an image in the Upload tab to see results here
                      </p>
                    </div>
                    <Button onClick={() => setActiveTab("upload")}>
                      <Upload className="h-4 w-4 mr-2" />
                      Go to Upload
                    </Button>
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          {/* History Tab */}
          <TabsContent value="history" className="space-y-6">
            <div className="space-y-2">
              <h2 className="text-2xl font-bold">Measurement History</h2>
              <p className="text-muted-foreground">View and manage your previous measurements</p>
            </div>

            <MeasurementHistory />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}