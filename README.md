# Kuramoto Synchronization Simulator

An advanced simulation platform that enables researchers and students to explore phase synchronization dynamics through interactive, visually rich network visualizations.

## Project Structure

The project follows a modern Python package structure:

```
kuramoto-simulator/
├── src/                      # Source code
│   ├── components/           # Modular components
│   ├── utils/                # Utility functions
│   ├── styles/               # CSS and styling
│   ├── models/               # Data models including Kuramoto model
│   ├── database/             # Database functionality
│   └── app.py                # Main application file
├── static/                   # Static assets
│   ├── images/               # Image files
│   └── json/                 # JSON configuration files
├── tests/                    # Test files
├── backups/                  # Backup files
├── .streamlit/               # Streamlit configuration
└── pyproject.toml            # Python project configuration
```

## Features

- Interactive visualization of oscillator synchronization
- Multiple network topologies: All-to-All, Ring, Random, Small-World, Scale-Free, and custom
- Frequency distribution options: Normal, Uniform, Golden Ratio, and custom
- Real-time animation of phase synchronization
- Order parameter analysis
- JSON configuration import/export
- Responsive design with gradient-based UI
- Time step optimization feature

## Technologies Used

- Python
- Streamlit
- Matplotlib
- NumPy
- NetworkX
- Plotly

## Running the Application

To run the application locally:

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```
   streamlit run src/app.py --server.port 5000
   ```

## Development

### Styling

The application uses custom CSS styles located in `src/styles/main.css`. The UI features gradient-based components and a responsive design.

### Adding New Features

When adding new features:
1. Maintain separation of concerns by placing functionality in appropriate directories
2. Update imports to reflect the modular structure
3. Follow the existing styling patterns

## Database

The application uses SQLAlchemy for database operations. The database schema and operations are defined in `src/database/database.py`.

## Testing

Tests are located in the `tests/` directory. Run tests using:

```
pytest tests/
```

## License

This project is open source and available under the MIT License.