import React, { useState } from 'react';
import { ChevronDown, ChevronUp, Camera, Cpu, Car, Database, Code, BarChart, Brain, GitBranch, Activity } from 'lucide-react';

const TeslaAutopilotArchitecture = () => {
  const [expandedSection, setExpandedSection] = useState(null);

  const toggleSection = (section) => {
    if (expandedSection === section) {
      setExpandedSection(null);
    } else {
      setExpandedSection(section);
    }
  };

  // Section components with expandable details
  const SectionCard = ({ id, title, icon, children }) => {
    const isExpanded = expandedSection === id;
    return (
      <div className="bg-white rounded-lg shadow-md border border-gray-200 w-full mb-4">
        <div 
          className="flex items-center justify-between p-4 cursor-pointer" 
          onClick={() => toggleSection(id)}
        >
          <div className="flex items-center">
            {icon}
            <h3 className="text-lg font-semibold ml-2">{title}</h3>
          </div>
          {isExpanded ? <ChevronUp className="text-gray-600" /> : <ChevronDown className="text-gray-600" />}
        </div>
        {isExpanded && (
          <div className="p-4 pt-0 border-t border-gray-200">
            {children}
          </div>
        )}
      </div>
    );
  };

  // Arrow component for the flow diagram
  const Arrow = ({ direction = "down", className = "" }) => {
    let transform = "";
    
    switch (direction) {
      case "right":
        transform = "rotate(90deg)";
        break;
      case "left":
        transform = "rotate(-90deg)";
        break;
      case "up":
        transform = "rotate(180deg)";
        break;
      default:
        transform = "";
    }
    
    return (
      <div className={`flex justify-center my-2 ${className}`}>
        <svg width="24" height="24" viewBox="0 0 24 24" style={{ transform }}>
          <path
            fill="currentColor"
            d="M12 4l-1.41 1.41L16.17 11H4v2h12.17l-5.58 5.59L12 20l8-8z"
          />
        </svg>
      </div>
    );
  };

  return (
    <div className="bg-gray-50 p-6 rounded-xl max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold mb-6 text-center">Tesla End-to-End Autopilot Architecture</h1>
      
      {/* Main architecture diagram */}
      <div className="flex flex-col items-center mb-8">
        <div className="w-full flex justify-center">
          <div className="bg-blue-600 text-white rounded-lg p-4 text-center max-w-md">
            <h2 className="font-bold">End-to-End Neural Network</h2>
            <p className="text-sm">Direct mapping from raw sensor inputs to vehicle control outputs</p>
          </div>
        </div>
        
        <Arrow />
        
        <div className="flex w-full justify-between">
          <div className="flex flex-col items-center">
            <div className="bg-purple-600 text-white rounded-lg p-3 text-center">
              <h3 className="font-semibold">Inputs</h3>
              <p className="text-xs">Camera, Radar, Ultrasonics</p>
            </div>
          </div>
          
          <div className="flex flex-col items-center">
            <div className="bg-green-600 text-white rounded-lg p-3 text-center">
              <h3 className="font-semibold">Outputs</h3>
              <p className="text-xs">Steering, Acceleration, Braking</p>
            </div>
          </div>
        </div>
      </div>
      
      {/* Detailed sections */}
      <div className="space-y-2">
        <SectionCard 
          id="data-collection" 
          title="Data Collection & Fleet Learning" 
          icon={<Database size={20} className="text-blue-500" />}
        >
          <div className="space-y-3">
            <p className="text-sm">Tesla vehicles continuously collect driving data from their sensor suite during both manual and autopilot operation.</p>
            
            <div className="bg-gray-100 p-3 rounded-md">
              <h4 className="font-medium text-sm mb-2">Data Collection Pipeline:</h4>
              <ul className="text-xs space-y-2">
                <li className="flex items-start">
                  <span className="bg-blue-100 text-blue-800 rounded-full px-2 py-1 mr-2">1</span>
                  <span>Camera feeds (8 cameras capturing 360° view)</span>
                </li>
                <li className="flex items-start">
                  <span className="bg-blue-100 text-blue-800 rounded-full px-2 py-1 mr-2">2</span>
                  <span>Radar and ultrasonic sensor data</span>
                </li>
                <li className="flex items-start">
                  <span className="bg-blue-100 text-blue-800 rounded-full px-2 py-1 mr-2">3</span>
                  <span>Vehicle telemetry (speed, acceleration, steering angle)</span>
                </li>
                <li className="flex items-start">
                  <span className="bg-blue-100 text-blue-800 rounded-full px-2 py-1 mr-2">4</span>
                  <span>Human driver interventions and corrections</span>
                </li>
                <li className="flex items-start">
                  <span className="bg-blue-100 text-blue-800 rounded-full px-2 py-1 mr-2">5</span>
                  <span>Edge cases and near-miss scenarios automatically flagged</span>
                </li>
              </ul>
            </div>
            
            <div className="flex justify-center">
              <div className="flex items-center">
                <div className="bg-gray-200 rounded p-2">
                  <Car size={16} />
                </div>
                <Arrow direction="right" className="mx-2" />
                <div className="bg-gray-200 rounded p-2">
                  <Database size={16} />
                </div>
              </div>
            </div>
          </div>
        </SectionCard>
        
        <SectionCard 
          id="data-processing" 
          title="Data Processing & Augmentation" 
          icon={<Code size={20} className="text-purple-500" />}
        >
          <div className="space-y-3">
            <p className="text-sm">Raw sensor data undergoes preprocessing to prepare it for neural network training.</p>
            
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-gray-100 p-3 rounded-md">
                <h4 className="font-medium text-sm mb-2">Preprocessing:</h4>
                <ul className="text-xs space-y-1">
                  <li>• Frame synchronization</li>
                  <li>• Sensor calibration</li>
                  <li>• Normalization</li>
                  <li>• Temporal alignment</li>
                </ul>
              </div>
              
              <div className="bg-gray-100 p-3 rounded-md">
                <h4 className="font-medium text-sm mb-2">Augmentation:</h4>
                <ul className="text-xs space-y-1">
                  <li>• Brightness/contrast variation</li>
                  <li>• Weather condition simulation</li>
                  <li>• Occlusion handling</li>
                  <li>• Geometric transformations</li>
                </ul>
              </div>
            </div>
            
            <div className="flex justify-center">
              <div className="bg-purple-100 p-2 rounded-md text-xs text-center">
                <p>Automated data labeling via self-supervised learning</p>
                <p className="font-semibold mt-1">1M+ frames processed daily</p>
              </div>
            </div>
          </div>
        </SectionCard>
        
        <SectionCard 
          id="network-architecture" 
          title="Neural Network Architecture" 
          icon={<Brain size={20} className="text-red-500" />}
        >
          <div className="space-y-3">
            <p className="text-sm">The end-to-end architecture directly maps raw sensor inputs to control outputs without explicit intermediate representations.</p>
            
            <div className="bg-gray-100 p-3 rounded-md">
              <h4 className="font-medium text-sm mb-2">Key Components:</h4>
              <ul className="text-xs space-y-2">
                <li className="flex items-start">
                  <div className="bg-red-100 text-red-800 rounded px-2 py-1 mr-2 w-20 text-center">
                    Video Encoder
                  </div>
                  <span>Processes multi-camera inputs using 3D convolutional networks</span>
                </li>
                <li className="flex items-start">
                  <div className="bg-red-100 text-red-800 rounded px-2 py-1 mr-2 w-20 text-center">
                    Transformer
                  </div>
                  <span>Self-attention mechanisms to model spatial and temporal relationships</span>
                </li>
                <li className="flex items-start">
                  <div className="bg-red-100 text-red-800 rounded px-2 py-1 mr-2 w-20 text-center">
                    Fusion Layer
                  </div>
                  <span>Combines visual features with radar and vehicle state</span>
                </li>
                <li className="flex items-start">
                  <div className="bg-red-100 text-red-800 rounded px-2 py-1 mr-2 w-20 text-center">
                    Policy Head
                  </div>
                  <span>Generates continuous control outputs (steering, acceleration, braking)</span>
                </li>
              </ul>
            </div>
            
            <div className="flex justify-center">
              <div className="bg-red-50 p-3 rounded-md text-xs">
                <p className="font-semibold text-center mb-2">Network Specifications</p>
                <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                  <div>Parameters:</div><div className="font-medium">1B+</div>
                  <div>Architecture:</div><div className="font-medium">Hybrid CNN-Transformer</div>
                  <div>Input frames:</div><div className="font-medium">8 cameras × 36fps</div>
                  <div>Temporal context:</div><div className="font-medium">2-3 seconds</div>
                </div>
              </div>
            </div>
          </div>
        </SectionCard>
        
        <SectionCard 
          id="training" 
          title="Training Methodology" 
          icon={<BarChart size={20} className="text-yellow-500" />}
        >
          <div className="space-y-3">
            <p className="text-sm">The end-to-end model is trained using multiple complementary approaches.</p>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="bg-yellow-50 p-3 rounded-md">
                <h4 className="font-medium text-sm mb-2 text-yellow-800">Imitation Learning</h4>
                <p className="text-xs">Learning from human demonstrations captured by the fleet</p>
                <ul className="text-xs mt-2 space-y-1">
                  <li>• Loss: MSE between predicted and human actions</li>
                  <li>• Source: Millions of driving hours</li>
                  <li>• Handles: Normal driving scenarios</li>
                </ul>
              </div>
              
              <div className="bg-yellow-50 p-3 rounded-md">
                <h4 className="font-medium text-sm mb-2 text-yellow-800">Reinforcement Learning</h4>
                <p className="text-xs">Optimizing actions beyond human demonstrations</p>
                <ul className="text-xs mt-2 space-y-1">
                  <li>• Reward: Safety, comfort, progress</li>
                  <li>• Environment: Simulation + real-world</li>
                  <li>• Handles: Complex scenarios, optimization</li>
                </ul>
              </div>
              
              <div className="bg-yellow-50 p-3 rounded-md">
                <h4 className="font-medium text-sm mb-2 text-yellow-800">Sim-to-Real</h4>
                <p className="text-xs">Training on simulated edge cases</p>
                <ul className="text-xs mt-2 space-y-1">
                  <li>• Source: Photorealistic simulations</li>
                  <li>• Focus: Rare and dangerous scenarios</li>
                  <li>• Handles: Edge cases not seen in fleet</li>
                </ul>
              </div>
            </div>
            
            <div className="flex justify-center">
              <div className="bg-yellow-100 p-2 rounded-md text-xs text-center">
                <p className="font-semibold">Distributed training across custom AI supercomputer</p>
                <p>Continuous training with daily data updates from the fleet</p>
              </div>
            </div>
          </div>
        </SectionCard>
        
        <SectionCard 
          id="deployment" 
          title="Deployment & Feedback Loop" 
          icon={<GitBranch size={20} className="text-green-500" />}
        >
          <div className="space-y-3">
            <p className="text-sm">Trained models are deployed to Tesla vehicles with continuous improvement through fleet data.</p>
            
            <div className="bg-gray-100 p-3 rounded-md">
              <h4 className="font-medium text-sm mb-2">Deployment Pipeline:</h4>
              <ul className="text-xs space-y-2">
                <li className="flex items-start">
                  <span className="bg-green-100 text-green-800 rounded-full px-2 py-1 mr-2">1</span>
                  <span>Shadow testing of new models against real-world data</span>
                </li>
                <li className="flex items-start">
                  <span className="bg-green-100 text-green-800 rounded-full px-2 py-1 mr-2">2</span>
                  <span>Quantization and optimization for FSD hardware</span>
                </li>
                <li className="flex items-start">
                  <span className="bg-green-100 text-green-800 rounded-full px-2 py-1 mr-2">3</span>
                  <span>A/B testing with limited fleet deployment</span>
                </li>
                <li className="flex items-start">
                  <span className="bg-green-100 text-green-800 rounded-full px-2 py-1 mr-2">4</span>
                  <span>Full rollout via over-the-air updates</span>
                </li>
              </ul>
            </div>
            
            <div className="flex justify-center items-center">
              <div className="flex flex-col items-center">
                <div className="bg-gray-200 rounded p-2">
                  <Car size={16} />
                </div>
                <div className="text-xs mt-1">Vehicle</div>
              </div>
              <Arrow direction="left" className="mx-2 transform rotate-180" />
              <div className="flex flex-col items-center">
                <div className="bg-gray-200 rounded p-2">
                  <Cpu size={16} />
                </div>
                <div className="text-xs mt-1">FSD Computer</div>
              </div>
              <Arrow direction="left" className="mx-2 transform rotate-180" />
              <div className="flex flex-col items-center">
                <div className="bg-gray-200 rounded p-2">
                  <Brain size={16} />
                </div>
                <div className="text-xs mt-1">Neural Network</div>
              </div>
            </div>
            
            <div className="bg-green-50 p-3 rounded-md">
              <h4 className="font-medium text-sm mb-2 text-center">Continuous Improvement Cycle</h4>
              <div className="flex justify-center">
                <div className="flex items-center">
                  <div className="text-xs text-center">
                    <div>Deploy</div>
                    <div>Model</div>
                  </div>
                  <Arrow direction="right" className="mx-2" />
                  <div className="text-xs text-center">
                    <div>Collect</div>
                    <div>Data</div>
                  </div>
                  <Arrow direction="right" className="mx-2" />
                  <div className="text-xs text-center">
                    <div>Analyze</div>
                    <div>Performance</div>
                  </div>
                  <Arrow direction="right" className="mx-2" />
                  <div className="text-xs text-center">
                    <div>Improve</div>
                    <div>Model</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </SectionCard>
        
        <SectionCard 
          id="hardware" 
          title="Hardware Acceleration" 
          icon={<Cpu size={20} className="text-gray-500" />}
        >
          <div className="space-y-3">
            <p className="text-sm">Custom-designed hardware optimized for neural network inference.</p>
            
            <div className="bg-gray-100 p-3 rounded-md">
              <h4 className="font-medium text-sm mb-2">FSD Computer:</h4>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div>• 2x Neural Processing Units</div>
                <div>• 144 TOPS computing power</div>
                <div>• Redundant design for safety</div>
                <div>• Low power consumption (72W)</div>
                <div>• Specialized matrix operations</div>
                <div>• Hardware-level security</div>
              </div>
            </div>
            
            <div className="flex justify-center">
              <div className="bg-gray-200 p-3 rounded-md text-xs">
                <p className="font-semibold text-center">Real-time Performance Metrics</p>
                <div className="grid grid-cols-2 gap-x-4 gap-y-1 mt-2">
                  <div>Inference time:</div><div className="font-medium">~10ms per frame</div>
                  <div>Frame processing:</div><div className="font-medium">36 fps</div>
                  <div>Parallel cameras:</div><div className="font-medium">8 simultaneous</div>
                  <div>Neural operations:</div><div className="font-medium">~1 trillion per second</div>
                </div>
              </div>
            </div>
          </div>
        </SectionCard>
        
        <SectionCard 
          id="monitoring" 
          title="Performance Monitoring" 
          icon={<Activity size={20} className="text-indigo-500" />}
        >
          <div className="space-y-3">
            <p className="text-sm">Continuous evaluation of model performance across the fleet.</p>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div className="bg-indigo-50 p-3 rounded-md">
                <h4 className="font-medium text-sm mb-2">Metrics Tracked:</h4>
                <ul className="text-xs space-y-1">
                  <li>• Disengagement frequency</li>
                  <li>• Prediction accuracy</li>
                  <li>• Control smoothness</li>
                  <li>• Intervention severity</li>
                  <li>• Geographic performance variation</li>
                  <li>• Weather condition robustness</li>
                </ul>
              </div>
              
              <div className="bg-indigo-50 p-3 rounded-md">
                <h4 className="font-medium text-sm mb-2">Evaluation Methods:</h4>
                <ul className="text-xs space-y-1">
                  <li>• A/B testing of model versions</li>
                  <li>• Simulation regression testing</li>
                  <li>• Closed-course validation</li>
                  <li>• Fleet telemetry analysis</li>
                  <li>• Intervention clustering</li>
                  <li>• Counterfactual simulation</li>
                </ul>
              </div>
            </div>
            
            <div className="flex justify-center">
              <div className="bg-indigo-100 p-2 rounded-md text-xs">
                <p className="font-semibold text-center">Automated Detection of Edge Cases</p>
                <p className="text-center">Continuous prioritization of challenging scenarios for targeted improvement</p>
              </div>
            </div>
          </div>
        </SectionCard>
      </div>
      
      {/* Legend/Footer */}
      <div className="mt-8 bg-white p-4 rounded-lg border border-gray-200">
        <h3 className="text-sm font-semibold mb-2">Key Advantages of End-to-End Architecture:</h3>
        <ul className="text-xs space-y-1">
          <li>• Eliminates hand-engineered components and rules</li>
          <li>• Learns directly from real-world data instead of programmed behaviors</li>
          <li>• Improves with more data without requiring algorithm changes</li>
          <li>• Reduces sensor fusion complexity by learning optimal integration</li>
          <li>• Enables holistic optimization of the entire driving stack</li>
        </ul>
      </div>
    </div>
  );
};

export default TeslaAutopilotArchitecture;