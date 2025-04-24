# Kuramoto Model Simulator

An advanced simulation platform for the Kuramoto model that enables researchers and students to explore phase synchronization dynamics through interactive, visually rich network visualizations.

## Project Structure

```
kuramoto-simulator/
├── README.md                 # Project documentation
├── src/                      # Source code
│   ├── components/           # Modular components
│   │   └── __init__.py       # Make folder a package
│   ├── utils/                # Utility functions
│   │   ├── __init__.py       # Make folder a package
│   │   └── ml_helper.py      # ML analytics utilities
│   ├── styles/               # CSS and styling
│   │   └── main.css          # Main stylesheet
│   ├── models/               # Data models
│   │   ├── __init__.py       # Make folder a package
│   │   └── kuramoto_model.py # Kuramoto model implementation
│   ├── database/             # Database functionality
│   │   ├── __init__.py       # Make folder a package
│   │   └── database.py       # Database operations
│   └── app.py                # Main application file
├── static/                   # Static assets
│   ├── images/               # Image files
│   │   └── wisp.jpg          # Background image
│   └── json/                 # JSON configuration files
│       └── default_config.json # Default configuration
├── tests/                    # Test files
├── backups/                  # Backup files
└── .streamlit/               # Streamlit configuration
    └── config.toml           # Streamlit config
```

## Features

- Interactive Kuramoto model simulation with real-time parameter adjustment
- Network visualization showing oscillator connectivity and synchronization
- Custom adjacency matrix input for specialized network topologies
- Time step optimization for numerical stability
- Frequency distribution visualization and analysis
- Order parameter time series analysis
- JSON configuration import/export
- Database storage for simulation results
- Phase animation for visualizing oscillator movement
- Gradient-based UI with responsive design

## Technologies Used

- Python
- Streamlit
- Matplotlib
- NumPy
- NetworkX
- SQLAlchemy
- SciPy
- Plotly

## Running the Application

To run the application locally:

```bash
streamlit run src/app.py
```

For deployment:

```bash
streamlit run src/app.py --server.port 5000 --server.address 0.0.0.0
```

## Development

The application follows a modular structure with separation of concerns:

- `models`: Contains the mathematical models and simulation logic
- `database`: Handles data persistence and retrieval
- `utils`: Utility functions and helpers
- `components`: Reusable UI components
- `static`: Static assets like images and default configurations

## License

[MIT License](LICENSE)