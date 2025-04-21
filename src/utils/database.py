"""
Database module for the Kuramoto Model Simulator.
This module provides functionality to store and retrieve simulation data,
as well as import/export configurations as JSON files.

Enhanced with machine learning support for:
- Dataset generation and export
- Feature extraction for neural networks
- Batch simulation processing
- Time series analysis
"""

import os
import numpy as np
import json
import pickle
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, LargeBinary, ForeignKey, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Create the engine and base
DATABASE_URL = "sqlite:///kuramoto_simulations.db"
engine = create_engine(DATABASE_URL)
Base = declarative_base()

# Session factory
Session = sessionmaker(bind=engine)


class Simulation(Base):
    """Model representing a Kuramoto simulation run."""
    __tablename__ = "simulations"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now)
    n_oscillators = Column(Integer)
    coupling_strength = Column(Float)
    simulation_time = Column(Float)
    time_step = Column(Float)
    random_seed = Column(Integer)
    frequency_distribution = Column(String(50))
    frequency_params = Column(Text)  # JSON string of distribution parameters
    
    # Relationships
    frequencies = relationship("Frequency", back_populates="simulation", cascade="all, delete-orphan")
    phases = relationship("Phase", back_populates="simulation", cascade="all, delete-orphan")
    order_parameters = relationship("OrderParameter", back_populates="simulation", cascade="all, delete-orphan")
    adjacency_matrix = relationship("AdjacencyMatrix", back_populates="simulation", uselist=False, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Simulation(id={self.id}, oscillators={self.n_oscillators}, coupling={self.coupling_strength})>"


class Frequency(Base):
    """Model representing oscillator natural frequencies."""
    __tablename__ = "frequencies"

    id = Column(Integer, primary_key=True)
    simulation_id = Column(Integer, ForeignKey("simulations.id"))
    oscillator_index = Column(Integer)
    value = Column(Float)
    
    simulation = relationship("Simulation", back_populates="frequencies")

    def __repr__(self):
        return f"<Frequency(simulation_id={self.simulation_id}, oscillator={self.oscillator_index}, value={self.value})>"


class Phase(Base):
    """Model representing phase data for each oscillator over time."""
    __tablename__ = "phases"

    id = Column(Integer, primary_key=True)
    simulation_id = Column(Integer, ForeignKey("simulations.id"))
    time_index = Column(Integer)
    oscillator_index = Column(Integer)
    value = Column(Float)
    
    simulation = relationship("Simulation", back_populates="phases")

    def __repr__(self):
        return f"<Phase(simulation_id={self.simulation_id}, time={self.time_index}, oscillator={self.oscillator_index}, value={self.value})>"


class OrderParameter(Base):
    """Model representing order parameter data over time."""
    __tablename__ = "order_parameters"

    id = Column(Integer, primary_key=True)
    simulation_id = Column(Integer, ForeignKey("simulations.id"))
    time_index = Column(Integer)
    magnitude = Column(Float)
    phase = Column(Float)
    
    simulation = relationship("Simulation", back_populates="order_parameters")

    def __repr__(self):
        return f"<OrderParameter(simulation_id={self.simulation_id}, time={self.time_index}, magnitude={self.magnitude})>"


class AdjacencyMatrix(Base):
    """Model representing the network adjacency matrix."""
    __tablename__ = "adjacency_matrices"

    id = Column(Integer, primary_key=True)
    simulation_id = Column(Integer, ForeignKey("simulations.id"), unique=True)
    data = Column(LargeBinary)  # Serialized numpy array
    network_type = Column(String(50))
    
    simulation = relationship("Simulation", back_populates="adjacency_matrix")

    def __repr__(self):
        return f"<AdjacencyMatrix(simulation_id={self.simulation_id}, network_type={self.network_type})>"


class Configuration(Base):
    """Model for storing and retrieving simulation configurations."""
    __tablename__ = "configurations"

    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    timestamp = Column(DateTime, default=datetime.now)
    n_oscillators = Column(Integer)
    coupling_strength = Column(Float)
    simulation_time = Column(Float)
    time_step = Column(Float)
    random_seed = Column(Integer)
    network_type = Column(String(50))
    frequency_distribution = Column(String(50))
    frequency_params = Column(Text)  # JSON string
    adjacency_matrix = Column(LargeBinary, nullable=True)  # Serialized numpy array

    def __repr__(self):
        return f"<Configuration(id={self.id}, name='{self.name}', oscillators={self.n_oscillators})>"


class MLDataset(Base):
    """Model for machine learning datasets composed of multiple simulations."""
    __tablename__ = "ml_datasets"

    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    description = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.now)
    feature_type = Column(String(50))  # e.g., 'time_series', 'graph', 'image'
    target_type = Column(String(50))   # e.g., 'regression', 'classification'
    
    simulations = relationship("MLDatasetSimulation", back_populates="dataset", cascade="all, delete-orphan")
    features = relationship("MLFeature", back_populates="dataset", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<MLDataset(id={self.id}, name='{self.name}', feature_type='{self.feature_type}')>"


class MLDatasetSimulation(Base):
    """Association between ML datasets and simulations."""
    __tablename__ = "ml_dataset_simulations"

    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey("ml_datasets.id"))
    simulation_id = Column(Integer, ForeignKey("simulations.id"))
    split = Column(String(20))  # 'train', 'validation', 'test'
    
    dataset = relationship("MLDataset", back_populates="simulations")
    simulation = relationship("Simulation")

    def __repr__(self):
        return f"<MLDatasetSimulation(dataset_id={self.dataset_id}, simulation_id={self.simulation_id}, split='{self.split}')>"


class MLFeature(Base):
    """Features extracted from simulations for machine learning."""
    __tablename__ = "ml_features"

    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey("ml_datasets.id"))
    name = Column(String(100))
    feature_type = Column(String(20))  # 'input', 'target', 'metadata'
    description = Column(Text, nullable=True)
    extraction_method = Column(Text)  # How the feature was extracted
    data = Column(LargeBinary)  # Serialized feature data
    
    dataset = relationship("MLDataset", back_populates="features")

    def __repr__(self):
        return f"<MLFeature(id={self.id}, name='{self.name}', feature_type='{self.feature_type}')>"


# Create tables
Base.metadata.create_all(engine)


def store_simulation(model, times, phases, order_parameter, frequencies, freq_type, freq_params=None, adjacency_matrix=None):
    """
    Store simulation results in the database.
    
    Parameters:
    -----------
    model : KuramotoModel
        The model instance containing simulation parameters
    times : ndarray
        Array of time points
    phases : ndarray
        Array of phase data (shape: time points × oscillators)
    order_parameter : dict
        Dictionary containing 'r' and 'psi' for order parameter magnitude and phase
    frequencies : ndarray
        Array of natural frequencies for each oscillator
    freq_type : str
        Type of frequency distribution (e.g., 'normal', 'uniform', 'custom')
    freq_params : dict, optional
        Parameters for the frequency distribution
    adjacency_matrix : ndarray, optional
        Adjacency matrix defining the network structure
        
    Returns:
    --------
    int
        ID of the stored simulation
    """
    try:
        # Start database session
        session = Session()
        
        # Create new simulation record
        sim = Simulation(
            n_oscillators=model.n_oscillators,
            coupling_strength=model.coupling_strength,
            simulation_time=model.simulation_time,
            time_step=model.time_step,
            random_seed=model.random_seed,
            frequency_distribution=freq_type,
            frequency_params=json.dumps(freq_params) if freq_params else None
        )
        session.add(sim)
        session.flush()  # Get ID without committing
        
        # Store natural frequencies
        for i, freq in enumerate(frequencies):
            freq_obj = Frequency(
                simulation_id=sim.id,
                oscillator_index=i,
                value=float(freq)
            )
            session.add(freq_obj)
        
        # Store phase data (sampled to prevent excessive storage)
        # For long simulations, we'll store fewer points to save space
        n_times = len(times)
        if n_times <= 100:
            # Store all time points if there are fewer than 100
            indices_to_store = range(n_times)
        else:
            # Store 100 evenly spaced points
            indices_to_store = np.linspace(0, n_times - 1, 100, dtype=int)
        
        # Store selected phase data
        for t_idx in indices_to_store:
            for osc_idx in range(model.n_oscillators):
                phase_obj = Phase(
                    simulation_id=sim.id,
                    time_index=t_idx,
                    oscillator_index=osc_idx,
                    value=float(phases[osc_idx, t_idx])
                )
                session.add(phase_obj)
        
        # Store order parameter data for all time points
        if isinstance(order_parameter, dict):
            # If order_parameter is a dictionary with r and psi
            r_values = order_parameter.get('r', [])
            psi_values = order_parameter.get('psi', [])
            
            for t_idx in indices_to_store:
                if t_idx < len(r_values):
                    order_obj = OrderParameter(
                        simulation_id=sim.id,
                        time_index=t_idx,
                        magnitude=float(r_values[t_idx]),
                        phase=float(psi_values[t_idx]) if len(psi_values) > t_idx else 0.0
                    )
                    session.add(order_obj)
        else:
            # If order_parameter is just an array of r values
            for t_idx in indices_to_store:
                if t_idx < len(order_parameter):
                    order_obj = OrderParameter(
                        simulation_id=sim.id,
                        time_index=t_idx,
                        magnitude=float(order_parameter[t_idx]),
                        phase=0.0  # Default if not provided
                    )
                    session.add(order_obj)
        
        # Store adjacency matrix if provided
        if adjacency_matrix is not None:
            # Determine network type based on structure
            network_type = "custom"  # Default
            adj_obj = AdjacencyMatrix(
                simulation_id=sim.id,
                data=pickle.dumps(adjacency_matrix),
                network_type=network_type
            )
            session.add(adj_obj)
        
        # Commit changes
        session.commit()
        return sim.id
    
    except Exception as e:
        print(f"Error storing simulation: {str(e)}")
        session.rollback()
        return None
    
    finally:
        session.close()


def get_simulation(simulation_id):
    """
    Retrieve a simulation from the database by ID.
    
    Parameters:
    -----------
    simulation_id : int
        ID of the simulation to retrieve
        
    Returns:
    --------
    dict
        Dictionary containing the simulation data, with keys:
        - 'params': Simulation parameters
        - 'times': Time points
        - 'phases': Phase data
        - 'order_parameter': Order parameter data
        - 'frequencies': Natural frequencies
        - 'adjacency_matrix': Network adjacency matrix (if available)
    """
    try:
        session = Session()
        
        # Get simulation record
        sim = session.query(Simulation).filter_by(id=simulation_id).first()
        if not sim:
            return None
        
        # Get frequencies
        frequencies = session.query(Frequency).filter_by(simulation_id=simulation_id).all()
        freq_values = []
        for freq in sorted(frequencies, key=lambda f: f.oscillator_index):
            freq_values.append(freq.value)
            
        # Get phase data
        phases = session.query(Phase).filter_by(simulation_id=simulation_id).all()
        
        # Determine unique time and oscillator indices
        time_indices = sorted(set(p.time_index for p in phases))
        oscillator_indices = sorted(set(p.oscillator_index for p in phases))
        
        # Create phase matrix
        phase_matrix = np.zeros((len(oscillator_indices), len(time_indices)))
        
        # Fill phase matrix
        for phase in phases:
            t_idx = time_indices.index(phase.time_index)
            osc_idx = oscillator_indices.index(phase.oscillator_index)
            phase_matrix[osc_idx, t_idx] = phase.value
            
        # Get order parameter data
        order_params = session.query(OrderParameter).filter_by(simulation_id=simulation_id).all()
        r_values = []
        psi_values = []
        
        for op in sorted(order_params, key=lambda o: o.time_index):
            r_values.append(op.magnitude)
            psi_values.append(op.phase)
            
        # Get adjacency matrix if available
        adj_matrix = session.query(AdjacencyMatrix).filter_by(simulation_id=simulation_id).first()
        adj_data = None
        
        if adj_matrix and adj_matrix.data:
            try:
                adj_data = pickle.loads(adj_matrix.data)
            except:
                pass  # Silently fail if pickle load fails
                
        # Parse frequency parameters if available
        freq_params = {}
        if sim.frequency_params:
            try:
                freq_params = json.loads(sim.frequency_params)
            except:
                pass
                
        # Construct result dictionary
        result = {
            'id': sim.id,
            'timestamp': sim.timestamp,
            'n_oscillators': sim.n_oscillators,
            'coupling_strength': sim.coupling_strength,
            'simulation_time': sim.simulation_time,
            'time_step': sim.time_step,
            'random_seed': sim.random_seed,
            'frequency_distribution': sim.frequency_distribution,
            'frequency_params': freq_params,
            'phases': phase_matrix,
            'times': np.array(range(len(time_indices))) * sim.time_step,
            'frequencies': np.array(freq_values),
            'order_parameter': {
                'r': np.array(r_values),
                'psi': np.array(psi_values)
            },
            'adjacency_matrix': adj_data
        }
        
        return result
    
    except Exception as e:
        print(f"Error retrieving simulation: {str(e)}")
        return None
    
    finally:
        session.close()


def list_simulations():
    """
    List all simulations in the database.
    
    Returns:
    --------
    list
        List of dictionaries, each containing metadata about a simulation
    """
    try:
        session = Session()
        simulations = session.query(Simulation).order_by(Simulation.timestamp.desc()).all()
        
        result = []
        for sim in simulations:
            # Simplified information for listing
            result.append({
                'id': sim.id,
                'timestamp': sim.timestamp,
                'n_oscillators': sim.n_oscillators,
                'coupling_strength': sim.coupling_strength,
                'frequency_distribution': sim.frequency_distribution
            })
            
        return result
    
    except Exception as e:
        print(f"Error listing simulations: {str(e)}")
        return []
    
    finally:
        session.close()


def delete_simulation(simulation_id):
    """
    Delete a simulation from the database.
    
    Parameters:
    -----------
    simulation_id : int
        ID of the simulation to delete
        
    Returns:
    --------
    bool
        True if deletion was successful, False otherwise
    """
    try:
        session = Session()
        sim = session.query(Simulation).filter_by(id=simulation_id).first()
        
        if sim:
            session.delete(sim)
            session.commit()
            return True
        
        return False
    
    except Exception as e:
        print(f"Error deleting simulation: {str(e)}")
        session.rollback()
        return False
    
    finally:
        session.close()


def save_configuration(name, n_oscillators, coupling_strength, simulation_time, time_step, 
                      random_seed, network_type, frequency_distribution, 
                      frequency_params=None, adjacency_matrix=None):
    """
    Save a simulation configuration to the database.
    
    Parameters:
    -----------
    name : str
        Name of the configuration
    n_oscillators : int
        Number of oscillators
    coupling_strength : float
        Coupling strength
    simulation_time : float
        Total simulation time
    time_step : float
        Simulation time step
    random_seed : int
        Random seed
    network_type : str
        Type of network ('all-to-all', 'custom', etc.)
    frequency_distribution : str
        Type of frequency distribution
    frequency_params : dict, optional
        Parameters for the frequency distribution
    adjacency_matrix : ndarray, optional
        Custom adjacency matrix
        
    Returns:
    --------
    int
        ID of the saved configuration
    """
    try:
        session = Session()
        
        # Check if configuration with this name already exists
        existing = session.query(Configuration).filter_by(name=name).first()
        if existing:
            # Update existing configuration
            existing.n_oscillators = n_oscillators
            existing.coupling_strength = coupling_strength
            existing.simulation_time = simulation_time
            existing.time_step = time_step
            existing.random_seed = random_seed
            existing.network_type = network_type
            existing.frequency_distribution = frequency_distribution
            existing.frequency_params = json.dumps(frequency_params) if frequency_params else None
            
            if adjacency_matrix is not None:
                existing.adjacency_matrix = pickle.dumps(adjacency_matrix)
                
            session.commit()
            return existing.id
        
        # Create new configuration
        config = Configuration(
            name=name,
            n_oscillators=n_oscillators,
            coupling_strength=coupling_strength,
            simulation_time=simulation_time,
            time_step=time_step,
            random_seed=random_seed,
            network_type=network_type,
            frequency_distribution=frequency_distribution,
            frequency_params=json.dumps(frequency_params) if frequency_params else None,
            adjacency_matrix=pickle.dumps(adjacency_matrix) if adjacency_matrix is not None else None
        )
        
        session.add(config)
        session.commit()
        
        return config.id
    
    except Exception as e:
        print(f"Error saving configuration: {str(e)}")
        session.rollback()
        return None
    
    finally:
        session.close()


def list_configurations():
    """
    List all saved configurations.
    
    Returns:
    --------
    list
        List of dictionaries, each containing metadata about a configuration
    """
    try:
        session = Session()
        configs = session.query(Configuration).order_by(Configuration.name).all()
        
        result = []
        for config in configs:
            # Get frequency parameters
            freq_params = {}
            if config.frequency_params:
                try:
                    freq_params = json.loads(config.frequency_params)
                except:
                    pass
                    
            result.append({
                'id': config.id,
                'name': config.name,
                'timestamp': config.timestamp,
                'n_oscillators': config.n_oscillators,
                'coupling_strength': config.coupling_strength,
                'simulation_time': config.simulation_time,
                'time_step': config.time_step,
                'random_seed': config.random_seed,
                'network_type': config.network_type,
                'frequency_distribution': config.frequency_distribution,
                'frequency_params': freq_params,
                'has_adjacency_matrix': config.adjacency_matrix is not None
            })
            
        return result
    
    except Exception as e:
        print(f"Error listing configurations: {str(e)}")
        return []
    
    finally:
        session.close()


def get_configuration(config_id):
    """
    Retrieve a configuration from the database by its ID.
    
    Parameters:
    -----------
    config_id : int
        ID of the configuration to retrieve
        
    Returns:
    --------
    dict
        Dictionary containing the configuration data
    """
    try:
        session = Session()
        config = session.query(Configuration).filter_by(id=config_id).first()
        
        if not config:
            return None
            
        # Get frequency parameters
        freq_params = {}
        if config.frequency_params:
            try:
                freq_params = json.loads(config.frequency_params)
            except:
                pass
                
        # Get adjacency matrix if available
        adj_matrix = None
        if config.adjacency_matrix:
            try:
                adj_matrix = pickle.loads(config.adjacency_matrix)
            except:
                pass
                
        # Construct result
        result = {
            'id': config.id,
            'name': config.name,
            'timestamp': config.timestamp,
            'n_oscillators': config.n_oscillators,
            'coupling_strength': config.coupling_strength,
            'simulation_time': config.simulation_time,
            'time_step': config.time_step,
            'random_seed': config.random_seed,
            'network_type': config.network_type,
            'frequency_distribution': config.frequency_distribution,
            'frequency_params': freq_params,
            'adjacency_matrix': adj_matrix
        }
        
        return result
    
    except Exception as e:
        print(f"Error retrieving configuration: {str(e)}")
        return None
    
    finally:
        session.close()


def get_configuration_by_name(name):
    """
    Retrieve a configuration from the database by its name.
    
    Parameters:
    -----------
    name : str
        Name of the configuration to retrieve
        
    Returns:
    --------
    dict
        Dictionary containing the configuration data
    """
    try:
        session = Session()
        config = session.query(Configuration).filter_by(name=name).first()
        
        if not config:
            return None
            
        # Redirect to the ID-based function
        return get_configuration(config.id)
    
    except Exception as e:
        print(f"Error retrieving configuration by name: {str(e)}")
        return None
    
    finally:
        session.close()


def delete_configuration(config_id):
    """
    Delete a configuration from the database.
    
    Parameters:
    -----------
    config_id : int
        ID of the configuration to delete
        
    Returns:
    --------
    bool
        True if deletion was successful, False otherwise
    """
    try:
        session = Session()
        config = session.query(Configuration).filter_by(id=config_id).first()
        
        if config:
            session.delete(config)
            session.commit()
            return True
            
        return False
    
    except Exception as e:
        print(f"Error deleting configuration: {str(e)}")
        session.rollback()
        return False
    
    finally:
        session.close()


def export_configuration_to_json(config_id, file_path=None):
    """
    Export a configuration as a JSON file.
    
    Parameters:
    -----------
    config_id : int
        ID of the configuration to export
    file_path : str, optional
        Path to save the JSON file. If None, returns the JSON string.
        
    Returns:
    --------
    str or None
        JSON string if file_path is None, otherwise None
    """
    try:
        # Get configuration
        config = get_configuration(config_id)
        if not config:
            return None
            
        # Convert to JSON-serializable format
        json_config = {
            'name': config['name'],
            'n_oscillators': config['n_oscillators'],
            'coupling_strength': config['coupling_strength'],
            'simulation_time': config['simulation_time'],
            'time_step': config['time_step'],
            'random_seed': config['random_seed'],
            'network_type': config['network_type'],
            'frequency_distribution': config['frequency_distribution'],
            'frequency_params': config['frequency_params']
        }
        
        # Handle adjacency matrix if present
        if config['adjacency_matrix'] is not None:
            # Convert numpy array to list for JSON serialization
            json_config['adjacency_matrix'] = config['adjacency_matrix'].tolist()
            
        # Convert to JSON string
        json_str = json.dumps(json_config, indent=4)
        
        # Save to file if path provided
        if file_path:
            with open(file_path, 'w') as f:
                f.write(json_str)
            return file_path
            
        # Otherwise return the JSON string
        return json_str
    
    except Exception as e:
        print(f"Error exporting configuration: {str(e)}")
        return None


def import_configuration_from_json(file_path, save_to_db=True):
    """
    Import a configuration from a JSON file and optionally save it to the database.
    
    Parameters:
    -----------
    file_path : str
        Path to the JSON configuration file
    save_to_db : bool
        Whether to save the imported configuration to the database
        
    Returns:
    --------
    dict or int
        Dictionary containing configuration data, or the ID of the saved configuration if save_to_db=True
    """
    try:
        # Read JSON file
        with open(file_path, 'r') as f:
            config = json.load(f)
            
        # Check for required fields
        required_fields = ['name', 'n_oscillators', 'coupling_strength', 
                           'simulation_time', 'time_step', 'network_type', 
                           'frequency_distribution']
                           
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field in JSON configuration: {field}")
                
        # Handle adjacency matrix if present
        if 'adjacency_matrix' in config and config['adjacency_matrix']:
            # Convert from list to numpy array
            config['adjacency_matrix'] = np.array(config['adjacency_matrix'])
            
        # Ensure random seed is present
        if 'random_seed' not in config or config['random_seed'] is None:
            config['random_seed'] = 42  # Default
            
        # Save to database if requested
        if save_to_db:
            config_id = save_configuration(
                name=config['name'],
                n_oscillators=config['n_oscillators'],
                coupling_strength=config['coupling_strength'],
                simulation_time=config['simulation_time'],
                time_step=config['time_step'],
                random_seed=config['random_seed'],
                network_type=config['network_type'],
                frequency_distribution=config['frequency_distribution'],
                frequency_params=config.get('frequency_params'),
                adjacency_matrix=config.get('adjacency_matrix')
            )
            return config_id
            
        # Otherwise return the configuration data
        return config
    
    except Exception as e:
        print(f"Error importing configuration: {str(e)}")
        return None


def create_ml_dataset(name, description=None, feature_type='time_series', target_type='regression'):
    """
    Create a new machine learning dataset.
    
    Parameters:
    -----------
    name : str
        Name of the dataset
    description : str, optional
        Description of the dataset
    feature_type : str, optional
        Type of features in the dataset (e.g., 'time_series', 'graph')
    target_type : str, optional
        Type of target variable (e.g., 'regression', 'classification')
        
    Returns:
    --------
    int
        ID of the created dataset
    """
    try:
        session = Session()
        
        # Create new dataset
        dataset = MLDataset(
            name=name,
            description=description,
            feature_type=feature_type,
            target_type=target_type
        )
        
        session.add(dataset)
        session.commit()
        
        return dataset.id
    
    except Exception as e:
        print(f"Error creating ML dataset: {str(e)}")
        session.rollback()
        return None
    
    finally:
        session.close()


def add_simulation_to_dataset(dataset_id, simulation_id, split='train'):
    """
    Add an existing simulation to a machine learning dataset.
    
    Parameters:
    -----------
    dataset_id : int
        ID of the dataset
    simulation_id : int
        ID of the simulation to add
    split : str, optional
        Dataset split ('train', 'validation', 'test')
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        session = Session()
        
        # Check if dataset exists
        dataset = session.query(MLDataset).filter_by(id=dataset_id).first()
        if not dataset:
            print(f"Dataset {dataset_id} not found")
            return False
            
        # Check if simulation exists
        simulation = session.query(Simulation).filter_by(id=simulation_id).first()
        if not simulation:
            print(f"Simulation {simulation_id} not found")
            return False
            
        # Check if simulation already in dataset
        existing = session.query(MLDatasetSimulation).filter_by(
            dataset_id=dataset_id, simulation_id=simulation_id).first()
            
        if existing:
            # Update split if already exists
            existing.split = split
        else:
            # Add new association
            ds_sim = MLDatasetSimulation(
                dataset_id=dataset_id,
                simulation_id=simulation_id,
                split=split
            )
            session.add(ds_sim)
            
        session.commit()
        return True
    
    except Exception as e:
        print(f"Error adding simulation to dataset: {str(e)}")
        session.rollback()
        return False
    
    finally:
        session.close()


def extract_features(dataset_id, feature_config):
    """
    Extract features from simulations in a dataset.
    
    Parameters:
    -----------
    dataset_id : int
        ID of the dataset
    feature_config : dict
        Configuration for feature extraction, with keys:
        - feature_name: {
            'type': 'input' or 'target',
            'description': Description of the feature,
            'extraction': Method to extract the feature (path or function)
          }
        
    Returns:
    --------
    list
        List of extracted feature IDs
    """
    try:
        session = Session()
        
        # Get the dataset
        dataset = session.query(MLDataset).filter_by(id=dataset_id).first()
        if not dataset:
            print(f"Dataset {dataset_id} not found")
            return []
            
        # Get all simulations in the dataset
        sim_assocs = session.query(MLDatasetSimulation).filter_by(dataset_id=dataset_id).all()
        sim_ids = [assoc.simulation_id for assoc in sim_assocs]
        
        if not sim_ids:
            print(f"No simulations found in dataset {dataset_id}")
            return []
            
        # Extract features for each feature configuration
        feature_ids = []
        
        for feature_name, config in feature_config.items():
            # Get extraction method
            extraction_method = config.get('extraction')
            if not extraction_method:
                print(f"No extraction method specified for feature {feature_name}")
                continue
                
            feature_type = config.get('type', 'input')
            description = config.get('description', f"Feature: {feature_name}")
            
            # Extract feature data for all simulations
            feature_data = []
            
            for sim_id in sim_ids:
                # Get simulation data
                sim_data = get_simulation(sim_id)
                if not sim_data:
                    print(f"Couldn't retrieve simulation {sim_id}")
                    continue
                    
                # Extract feature based on extraction method
                try:
                    # For now we'll use a simple extraction method based on path notation
                    # e.g., order_parameter.r or params.coupling_strength
                    if extraction_method == 'params':
                        extracted = {
                            'value': sim_data['coupling_strength'],
                            'n_oscillators': sim_data['n_oscillators'],
                            'network_type': sim_data.get('network_type', 'all-to-all')
                        }
                    elif extraction_method == 'frequencies':
                        extracted = sim_data['frequencies']
                    elif extraction_method == 'adjacency_matrix':
                        extracted = sim_data.get('adjacency_matrix')
                    elif extraction_method == 'phases':
                        # Use the final phase configuration
                        extracted = sim_data['phases'][:, -1]
                    elif '.' in extraction_method:
                        # Use path notation (e.g., order_parameter.r)
                        obj_path, attr = extraction_method.split('.', 1)
                        if obj_path == 'order_parameter':
                            extracted = sim_data['order_parameter'][attr]
                        else:
                            print(f"Unknown object path: {obj_path}")
                            continue
                    else:
                        print(f"Unknown extraction method: {extraction_method}")
                        continue
                        
                    feature_data.append(extracted)
                    
                except Exception as e:
                    print(f"Error extracting feature {feature_name} from simulation {sim_id}: {str(e)}")
                    continue
                    
            # Check if we got any data
            if not feature_data:
                print(f"No feature data extracted for {feature_name}")
                continue
                
            # Store the feature data
            serialized_data = pickle.dumps(feature_data)
            
            # Create feature record
            feature = MLFeature(
                dataset_id=dataset_id,
                name=feature_name,
                feature_type=feature_type,
                description=description,
                extraction_method=extraction_method,
                data=serialized_data
            )
            
            session.add(feature)
            session.flush()
            feature_ids.append(feature.id)
            
        session.commit()
        return feature_ids
    
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        session.rollback()
        return []
    
    finally:
        session.close()


def get_ml_dataset(dataset_id):
    """
    Get a machine learning dataset with its features and simulations.
    
    Parameters:
    -----------
    dataset_id : int
        ID of the dataset
        
    Returns:
    --------
    dict
        Dictionary containing the dataset information
    """
    try:
        session = Session()
        
        # Get dataset
        dataset = session.query(MLDataset).filter_by(id=dataset_id).first()
        if not dataset:
            return None
            
        # Get simulations in the dataset
        sim_assocs = session.query(MLDatasetSimulation).filter_by(dataset_id=dataset_id).all()
        simulations = []
        
        for assoc in sim_assocs:
            sim = session.query(Simulation).filter_by(id=assoc.simulation_id).first()
            if sim:
                simulations.append({
                    'id': sim.id,
                    'split': assoc.split,
                    'n_oscillators': sim.n_oscillators,
                    'coupling_strength': sim.coupling_strength,
                    'frequency_distribution': sim.frequency_distribution
                })
                
        # Get features
        features = []
        feature_records = session.query(MLFeature).filter_by(dataset_id=dataset_id).all()
        
        for feature in feature_records:
            try:
                # Deserialize feature data
                data = pickle.loads(feature.data)
                
                features.append({
                    'id': feature.id,
                    'name': feature.name,
                    'feature_type': feature.feature_type,
                    'description': feature.description,
                    'extraction_method': feature.extraction_method,
                    'data': data
                })
                
            except Exception as e:
                print(f"Error deserializing feature {feature.id}: {str(e)}")
                
        # Construct result
        result = {
            'id': dataset.id,
            'name': dataset.name,
            'description': dataset.description,
            'timestamp': dataset.timestamp,
            'feature_type': dataset.feature_type,
            'target_type': dataset.target_type,
            'simulations': simulations,
            'features': features
        }
        
        return result
    
    except Exception as e:
        print(f"Error getting ML dataset: {str(e)}")
        return None
    
    finally:
        session.close()


def list_ml_datasets():
    """
    List all machine learning datasets.
    
    Returns:
    --------
    list
        List of dictionaries with dataset information
    """
    try:
        session = Session()
        datasets = session.query(MLDataset).all()
        
        result = []
        for dataset in datasets:
            # Get count of simulations
            sim_count = session.query(MLDatasetSimulation).filter_by(dataset_id=dataset.id).count()
            
            # Get count of features
            feature_count = session.query(MLFeature).filter_by(dataset_id=dataset.id).count()
            
            # Get simulations
            sim_assocs = session.query(MLDatasetSimulation).filter_by(dataset_id=dataset.id).all()
            simulations = []
            
            for assoc in sim_assocs:
                sim = session.query(Simulation).filter_by(id=assoc.simulation_id).first()
                if sim:
                    simulations.append({
                        'id': sim.id,
                        'split': assoc.split
                    })
                    
            # Get features (metadata only)
            features = []
            feature_records = session.query(MLFeature).filter_by(dataset_id=dataset.id).all()
            
            for feature in feature_records:
                features.append({
                    'id': feature.id,
                    'name': feature.name,
                    'feature_type': feature.feature_type,
                    'description': feature.description
                })
                
            result.append({
                'id': dataset.id,
                'name': dataset.name,
                'description': dataset.description,
                'timestamp': dataset.timestamp,
                'feature_type': dataset.feature_type,
                'target_type': dataset.target_type,
                'sim_count': sim_count,
                'feature_count': feature_count,
                'simulations': simulations,
                'features': features
            })
            
        return result
    
    except Exception as e:
        print(f"Error listing ML datasets: {str(e)}")
        return []
    
    finally:
        session.close()


def export_ml_dataset(dataset_id, export_dir=None, format='numpy', train_ratio=0.7, val_ratio=0.15):
    """
    Export a machine learning dataset for use with ML frameworks.
    
    Parameters:
    -----------
    dataset_id : int
        ID of the dataset to export
    export_dir : str, optional
        Directory to save the exported dataset. If None, a timestamped directory is created.
    format : str, optional
        Export format ('numpy', 'pandas', 'pickle')
    train_ratio : float, optional
        Ratio of data for training (if no split specified)
    val_ratio : float, optional
        Ratio of data for validation (if no split specified)
        
    Returns:
    --------
    str
        Path to the exported dataset
    """
    try:
        # Get dataset
        dataset = get_ml_dataset(dataset_id)
        if not dataset:
            print(f"Dataset {dataset_id} not found")
            return None
            
        # Create export directory
        if export_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_dir = f"ml_dataset_{dataset['name'].replace(' ', '_')}_{timestamp}"
            
        os.makedirs(export_dir, exist_ok=True)
        
        # Create split directories
        splits = ['train', 'validation', 'test']
        for split in splits:
            os.makedirs(os.path.join(export_dir, split), exist_ok=True)
            
        # Determine which simulations go in which split
        split_map = {}
        has_explicit_splits = any(sim['split'] for sim in dataset['simulations'])
        
        if has_explicit_splits:
            # Use explicitly defined splits
            for sim in dataset['simulations']:
                split = sim['split']
                if split not in split_map:
                    split_map[split] = []
                split_map[split].append(sim['id'])
        else:
            # Assign splits based on ratios
            all_sims = [sim['id'] for sim in dataset['simulations']]
            n_sims = len(all_sims)
            
            # Shuffle for random assignment
            import random
            random.shuffle(all_sims)
            
            # Calculate indices for splits
            train_end = int(n_sims * train_ratio)
            val_end = train_end + int(n_sims * val_ratio)
            
            split_map['train'] = all_sims[:train_end]
            split_map['validation'] = all_sims[train_end:val_end]
            split_map['test'] = all_sims[val_end:]
            
        # Export features
        for feature in dataset['features']:
            feature_name = feature['name']
            feature_data = feature['data']
            
            # For each split, get the feature data for the simulations in that split
            for split, sim_ids in split_map.items():
                if not sim_ids:
                    continue
                    
                # Filter data for this split
                split_indices = [i for i, sim in enumerate(dataset['simulations']) 
                               if sim['id'] in sim_ids]
                split_data = [feature_data[i] for i in split_indices if i < len(feature_data)]
                
                if not split_data:
                    continue
                    
                # Export based on format
                if format == 'numpy':
                    # Convert to numpy array (if not already)
                    try:
                        np_data = np.array(split_data)
                        np.save(os.path.join(export_dir, split, f"{feature_name}.npy"), np_data)
                    except:
                        # If conversion fails, save as pickle
                        with open(os.path.join(export_dir, split, f"{feature_name}.pkl"), 'wb') as f:
                            pickle.dump(split_data, f)
                            
                elif format == 'pandas':
                    # Try to convert to DataFrame
                    try:
                        import pandas as pd
                        df = pd.DataFrame(split_data)
                        df.to_csv(os.path.join(export_dir, split, f"{feature_name}.csv"), index=False)
                    except:
                        # If conversion fails, save as pickle
                        with open(os.path.join(export_dir, split, f"{feature_name}.pkl"), 'wb') as f:
                            pickle.dump(split_data, f)
                            
                elif format == 'pickle':
                    # Save as pickle
                    with open(os.path.join(export_dir, split, f"{feature_name}.pkl"), 'wb') as f:
                        pickle.dump(split_data, f)
                        
        # Save dataset metadata
        meta = {
            'id': dataset['id'],
            'name': dataset['name'],
            'description': dataset['description'],
            'feature_type': dataset['feature_type'],
            'target_type': dataset['target_type'],
            'features': [{'name': f['name'], 'type': f['feature_type']} for f in dataset['features']],
            'splits': {split: len(sims) for split, sims in split_map.items() if sims}
        }
        
        with open(os.path.join(export_dir, 'metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2, default=str)
            
        return os.path.abspath(export_dir)
    
    except Exception as e:
        print(f"Error exporting ML dataset: {str(e)}")
        return None


def run_batch_simulations(config_variations, base_config, dataset_name=None):
    """
    Run multiple simulations with varying parameters and optionally save to a dataset.
    
    Parameters:
    -----------
    config_variations : list
        List of dictionaries, each containing parameter variations
    base_config : dict
        Base configuration to use for all simulations
    dataset_name : str, optional
        Name of dataset to create for these simulations
        
    Returns:
    --------
    dict
        Dictionary with results of the batch run
    """
    try:
        from src.models.kuramoto_model import KuramotoModel
        
        # Create dataset if name provided
        dataset_id = None
        if dataset_name:
            dataset_id = create_ml_dataset(
                name=dataset_name,
                description=f"Batch run with {len(config_variations)} variations",
                feature_type="time_series",
                target_type="regression"
            )
            
        # Run simulations
        sim_ids = []
        for variation in config_variations:
            # Merge base config and variation
            config = base_config.copy()
            config.update(variation)
            
            # Extract parameters
            n_oscillators = config.get("n_oscillators", 10)
            coupling_strength = config.get("coupling_strength", 1.0)
            simulation_time = config.get("simulation_time", 20.0)
            time_step = config.get("time_step", 0.01)
            random_seed = config.get("random_seed", 42)
            
            # Create adjacency matrix
            network_type = config.get("network_type", "all-to-all")
            adjacency_matrix = None
            
            if network_type == "all-to-all":
                # Fully connected network
                adjacency_matrix = np.ones((n_oscillators, n_oscillators))
                np.fill_diagonal(adjacency_matrix, 0)
            elif network_type == "ring":
                # Ring topology
                adjacency_matrix = np.zeros((n_oscillators, n_oscillators))
                for i in range(n_oscillators):
                    adjacency_matrix[i, (i-1) % n_oscillators] = 1
                    adjacency_matrix[i, (i+1) % n_oscillators] = 1
            elif network_type == "random":
                # Random network with 20% connectivity
                np.random.seed(random_seed)
                adjacency_matrix = (np.random.random((n_oscillators, n_oscillators)) < 0.2).astype(float)
                np.fill_diagonal(adjacency_matrix, 0)
                
            # Generate frequencies based on distribution
            freq_distribution = config.get("frequency_distribution", "normal")
            freq_params = config.get("frequency_params", {})
            frequencies = None
            
            if freq_distribution == "normal":
                mean = freq_params.get("mean", 0.0)
                std = freq_params.get("std", 1.0)
                np.random.seed(random_seed)
                frequencies = np.random.normal(mean, std, n_oscillators)
            elif freq_distribution == "uniform":
                min_val = freq_params.get("min", -1.0)
                max_val = freq_params.get("max", 1.0)
                np.random.seed(random_seed)
                frequencies = np.random.uniform(min_val, max_val, n_oscillators)
            elif freq_distribution == "custom" and "values" in freq_params:
                frequencies = np.array(freq_params["values"])
                # Resize if needed
                if len(frequencies) != n_oscillators:
                    frequencies = np.resize(frequencies, n_oscillators)
                    
            # Run the simulation
            model = KuramotoModel(
                n_oscillators=n_oscillators,
                coupling_strength=coupling_strength,
                frequencies=frequencies,
                simulation_time=simulation_time,
                time_step=time_step,
                random_seed=random_seed,
                adjacency_matrix=adjacency_matrix
            )
            
            times, phases, order_parameter = model.simulate()
            
            # Store in database
            sim_id = store_simulation(
                model=model,
                times=times,
                phases=phases,
                order_parameter=order_parameter,
                frequencies=frequencies,
                freq_type=freq_distribution,
                freq_params=freq_params,
                adjacency_matrix=adjacency_matrix
            )
            
            if sim_id:
                sim_ids.append(sim_id)
                
                # Add to dataset if created
                if dataset_id:
                    add_simulation_to_dataset(dataset_id, sim_id)
                    
        # Return results
        return {
            "completed": len(sim_ids),
            "sim_ids": sim_ids,
            "dataset_id": dataset_id
        }
    
    except Exception as e:
        print(f"Error running batch simulations: {str(e)}")
        return {"error": str(e), "completed": 0}