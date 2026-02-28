# Forge — Graph-Native ML Workflow OS

Forge is a graph-native machine learning workflow operating system designed to streamline ML experiment management, versioning, and visualization.

## Architecture

### Backend
- **FastAPI**: Modern, fast web framework for building APIs
- **SQLAlchemy**: SQL toolkit and ORM for database operations
- **SQLite**: Lightweight database for development
- **Pydantic**: Data validation using Python type annotations

### Frontend
- **React + TypeScript**: Modern UI framework with type safety
- **Vite**: Fast build tool and development server
- **React Flow**: Library for building node-based UIs and interactive graphs
- **Zustand**: Small, fast, and scalable state management
- **TailwindCSS**: Utility-first CSS framework for styling

## Project Structure

```
forge/
├── backend/
│   ├── main.py              # FastAPI application entry point
│   ├── database.py          # Database configuration and session management
│   ├── models.py            # SQLAlchemy database models
│   ├── schemas.py           # Pydantic data schemas
│   ├── routers/             # API route handlers
│   │   ├── graphs.py        # Graph management endpoints
│   │   ├── experiments.py   # Experiment management endpoints
│   │   └── versions.py      # Version control endpoints
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── package.json         # Node.js dependencies and scripts
│   ├── src/
│   │   ├── App.tsx          # Main React application component
│   │   ├── main.tsx         # React application entry point
│   │   ├── components/      # React components
│   │   │   ├── GraphEditor.tsx      # Interactive graph editing interface
│   │   │   ├── LineageTree.tsx      # Experiment lineage visualization
│   │   │   ├── ExperimentDashboard.tsx  # Experiment management dashboard
│   │   │   └── DiffViewer.tsx       # Version comparison interface
│   │   ├── store/           # Zustand state management
│   │   │   └── useStore.ts  # Global application state
│   │   └── api/             # API client utilities
│   │       └── client.ts    # HTTP client for backend communication
└── README.md                # This file
```

## Getting Started

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Unix/macOS:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the FastAPI server:
   ```bash
   python -m uvicorn main:app --reload
   ```

The backend API will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

The frontend application will be available at `http://localhost:5173`

## API Endpoints

### Health Check
- `GET /health` - Returns server status

### Graphs
- `GET /graphs` - List all graphs
- `GET /graphs/{id}` - Get specific graph
- `POST /graphs` - Create new graph
- `PUT /graphs/{id}` - Update graph
- `DELETE /graphs/{id}` - Delete graph

### Experiments
- `GET /experiments` - List all experiments
- `GET /experiments/{id}` - Get specific experiment
- `POST /experiments` - Create new experiment
- `PUT /experiments/{id}` - Update experiment
- `DELETE /experiments/{id}` - Delete experiment

### Versions
- `GET /versions` - List all versions
- `GET /versions/{id}` - Get specific version
- `POST /versions` - Create new version
- `DELETE /versions/{id}` - Delete version

## Features Implemented

- **Visual Graph Editor** - Drag-and-drop node interface with React Flow  
- **Hyperparameter Sweep Orchestrator** - Parallel experiment execution with real-time monitoring  
- **Structural Diff Engine** - Compare model architectures across versions  
- **Lineage Tree** - Full version control and experiment tracking  
- **Impact Analysis** - Correlate architecture changes with performance metrics  
- **Custom Algorithm Upload** - Execute user-defined training algorithms  
- **Real Math-Based Training** - Gradient descent and Adam optimizer implementations  
- **Dark Theme UI** - Modern, responsive interface with TailwindCSS  
- **Async Backend** - FastAPI with async SQLAlchemy for high performance  

## License

[Add your license information here]
