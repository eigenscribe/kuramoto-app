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
    
    # Relationship back to simulation
    simulation = relationship("Simulation", back_populates="frequencies")

    def __repr__(self):
        return f"<Frequency(oscillator={self.oscillator_index}, value={self.value})>"


class Phase(Base):
    """Model representing phase data for each oscillator over time."""
    __tablename__ = "phases"

    id = Column(Integer, primary_key=True)
    simulation_id = Column(Integer, ForeignKey("simulations.id"))
    time_index = Column(Integer)
    oscillator_index = Column(Integer)
    value = Column(Float)
    
    # Relationship back to simulation
    simulation = relationship("Simulation", back_populates="phases")

    def __repr__(self):
        return f"<Phase(time={self.time_index}, oscillator={self.oscillator_index}, value={self.value})>"


class OrderParameter(Base):
    """Model representing order parameter data over time."""
    __tablename__ = "order_parameters"

    id = Column(Integer, primary_key=True)
    simulation_id = Column(Integer, ForeignKey("simulations.id"))
    time_index = Column(Integer)
    magnitude = Column(Float)
    phase = Column(Float)
    
    # Relationship back to simulation
    simulation = relationship("Simulation", back_populates="order_parameters")

    def __repr__(self):
        return f"<OrderParameter(time={self.time_index}, r={self.magnitude}, psi={self.phase})>"


class AdjacencyMatrix(Base):
    """Model representing the network adjacency matrix."""
    __tablename__ = "adjacency_matrices"

    id = Column(Integer, primary_key=True)
    simulation_id = Column(Integer, ForeignKey("simulations.id"), unique=True)
    data = Column(LargeBinary)  # Serialized numpy array
    network_type = Column(String(50))
    
    # Relationship back to simulation
    simulation = relationship("Simulation", back_populates="adjacency_matrix")

    def __repr__(self):
        return f"<AdjacencyMatrix(sim_id={self.simulation_id}, type={self.network_type})>"


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
        return f"<Configuration(id={self.id}, name={self.name})>"


class MLDataset(Base):
    """Model for machine learning datasets composed of multiple simulations."""
    __tablename__ = "ml_datasets"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    description = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.now)
    feature_type = Column(String(50))  # e.g., 'time_series', 'graph', 'image'
    target_type = Column(String(50))   # e.g., 'regression', 'classification'
    
    # Relationships
    simulations = relationship("MLDatasetSimulation", back_populates="dataset", cascade="all, delete-orphan")
    features = relationship("MLFeature", back_populates="dataset", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<MLDataset(id={self.id}, name={self.name})>"


class MLDatasetSimulation(Base):
    """Association between ML datasets and simulations."""
    __tablename__ = "ml_dataset_simulations"
    
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey("ml_datasets.id"))
    simulation_id = Column(Integer, ForeignKey("simulations.id"))
    split = Column(String(20))  # 'train', 'validation', 'test'
    
    # Relationships
    dataset = relationship("MLDataset", back_populates="simulations")
    simulation = relationship("Simulation")
    
    def __repr__(self):
        return f"<MLDatasetSimulation(dataset_id={self.dataset_id}, simulation_id={self.simulation_id})>"


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
    
    # Relationship
    dataset = relationship("MLDataset", back_populates="features")
    
    def __repr__(self):
        return f"<MLFeature(id={self.id}, name={self.name})>"


# Create all tables if they don't exist
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
    # Create a new session
    session = Session()
    
    try:
        # Convert adjacency matrix to binary if it exists
        adj_matrix_binary = None
        network_type = "All-to-All"
        
        if adjacency_matrix is not None:
            adj_matrix_binary = pickle.dumps(adjacency_matrix)
            network_type = "Custom Adjacency Matrix"
        
        # Convert frequency params to JSON if they exist
        freq_params_json = None
        if freq_params is not None:
            freq_params_json = json.dumps(freq_params)
        
        # Create the simulation record
        simulation = Simulation(
            n_oscillators=model.n_oscillators,
            coupling_strength=model.coupling_strength,
            simulation_time=model.simulation_time,
            time_step=model.time_step,
            random_seed=model.random_seed,
            frequency_distribution=freq_type,
            frequency_params=freq_params_json
        )
        
        # Add to session
        session.add(simulation)
        session.flush()  # Flush to get the ID
        
        # Store the adjacency matrix if it exists
        if adj_matrix_binary is not None:
            adj_matrix_record = AdjacencyMatrix(
                simulation_id=simulation.id,
                data=adj_matrix_binary,
                network_type=network_type
            )
            session.add(adj_matrix_record)
        
        # Store frequencies
        for i, freq in enumerate(frequencies):
            freq_record = Frequency(
                simulation_id=simulation.id,
                oscillator_index=i,
                value=float(freq)
            )
            session.add(freq_record)
        
        # Store phase data (if phases is large, store a subset)
        max_phase_samples = 1000
        if len(times) > max_phase_samples:
            # Sample phases at regular intervals
            sample_indices = np.linspace(0, len(times) - 1, max_phase_samples, dtype=int)
            times_sampled = times[sample_indices]
            phases_sampled = phases[sample_indices]
        else:
            times_sampled = times
            phases_sampled = phases
        
        # Store phases
        for t_idx, t in enumerate(times_sampled):
            for osc_idx in range(model.n_oscillators):
                phase_record = Phase(
                    simulation_id=simulation.id,
                    time_index=t_idx,
                    oscillator_index=osc_idx,
                    value=float(phases_sampled[t_idx, osc_idx])
                )
                session.add(phase_record)
        
        # Store order parameter (if order_parameter is large, store a subset)
        r_values = order_parameter['r']
        psi_values = order_parameter['psi']
        
        if len(r_values) > max_phase_samples:
            # Sample order parameter at regular intervals
            sample_indices = np.linspace(0, len(r_values) - 1, max_phase_samples, dtype=int)
            r_sampled = r_values[sample_indices]
            psi_sampled = psi_values[sample_indices]
        else:
            r_sampled = r_values
            psi_sampled = psi_values
        
        for t_idx, (r, psi) in enumerate(zip(r_sampled, psi_sampled)):
            order_param_record = OrderParameter(
                simulation_id=simulation.id,
                time_index=t_idx,
                magnitude=float(r),
                phase=float(psi)
            )
            session.add(order_param_record)
        
        # Commit the transaction
        session.commit()
        return simulation.id
    
    except Exception as e:
        session.rollback()
        print(f"Error storing simulation: {e}")
        raise
    
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
    # Create a session
    session = Session()
    
    try:
        # Query the simulation
        simulation = session.query(Simulation).get(simulation_id)
        
        if not simulation:
            return None
        
        # Retrieve frequencies
        frequencies = session.query(Frequency).filter_by(simulation_id=simulation_id).order_by(Frequency.oscillator_index).all()
        freq_values = np.array([f.value for f in frequencies])
        
        # Retrieve phases
        phases = session.query(Phase).filter_by(simulation_id=simulation_id).order_by(Phase.time_index, Phase.oscillator_index).all()
        time_indices = sorted(list(set([p.time_index for p in phases])))
        oscillator_indices = sorted(list(set([p.oscillator_index for p in phases])))
        
        # Create a phases array
        phases_array = np.zeros((len(time_indices), len(oscillator_indices)))
        for p in phases:
            phases_array[p.time_index, p.oscillator_index] = p.value
        
        # Retrieve order parameter
        order_params = session.query(OrderParameter).filter_by(simulation_id=simulation_id).order_by(OrderParameter.time_index).all()
        r_values = np.array([op.magnitude for op in order_params])
        psi_values = np.array([op.phase for op in order_params])
        
        # Retrieve adjacency matrix if available
        adj_matrix = session.query(AdjacencyMatrix).filter_by(simulation_id=simulation_id).first()
        adj_matrix_array = None
        network_type = "All-to-All"
        
        if adj_matrix:
            adj_matrix_array = pickle.loads(adj_matrix.data)
            network_type = adj_matrix.network_type
        
        # Parse frequency params if available
        freq_params = None
        if simulation.frequency_params:
            try:
                freq_params = json.loads(simulation.frequency_params)
            except:
                freq_params = simulation.frequency_params
        
        # Create a result dictionary
        result = {
            'params': {
                'id': simulation.id,
                'timestamp': simulation.timestamp,
                'n_oscillators': simulation.n_oscillators,
                'coupling_strength': simulation.coupling_strength,
                'simulation_time': simulation.simulation_time,
                'time_step': simulation.time_step,
                'random_seed': simulation.random_seed,
                'frequency_distribution': simulation.frequency_distribution,
                'frequency_params': freq_params,
                'network_type': network_type
            },
            'frequencies': freq_values,
            'times': np.array(time_indices),
            'phases': phases_array,
            'order_parameter': {
                'r': r_values,
                'psi': psi_values
            },
            'adjacency_matrix': adj_matrix_array
        }
        
        return result
    
    except Exception as e:
        print(f"Error retrieving simulation: {e}")
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
    # Create a session
    session = Session()
    
    try:
        # Query all simulations
        simulations = session.query(Simulation).order_by(Simulation.timestamp.desc()).all()
        
        # Create a list of result dictionaries
        results = []
        for sim in simulations:
            # Parse frequency params if available
            freq_params = None
            if sim.frequency_params:
                try:
                    freq_params = json.loads(sim.frequency_params)
                except:
                    freq_params = sim.frequency_params
            
            # Get the adjacency matrix record if available
            adj_matrix = session.query(AdjacencyMatrix).filter_by(simulation_id=sim.id).first()
            network_type = "All-to-All"
            if adj_matrix:
                network_type = adj_matrix.network_type
            
            # Add result to the list
            results.append({
                'id': sim.id,
                'timestamp': sim.timestamp,
                'n_oscillators': sim.n_oscillators,
                'coupling_strength': sim.coupling_strength,
                'simulation_time': sim.simulation_time,
                'time_step': sim.time_step,
                'random_seed': sim.random_seed,
                'frequency_distribution': sim.frequency_distribution,
                'frequency_params': freq_params,
                'network_type': network_type
            })
        
        return results
    
    except Exception as e:
        print(f"Error listing simulations: {e}")
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
    # Create a session
    session = Session()
    
    try:
        # Query the simulation
        simulation = session.query(Simulation).get(simulation_id)
        
        if not simulation:
            return False
        
        # Delete the simulation (cascade will handle related records)
        session.delete(simulation)
        session.commit()
        
        return True
    
    except Exception as e:
        session.rollback()
        print(f"Error deleting simulation: {e}")
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
    # Create a session
    session = Session()
    
    try:
        # Convert frequency params to JSON if they exist
        freq_params_json = None
        if frequency_params is not None:
            if isinstance(frequency_params, dict):
                freq_params_json = json.dumps(frequency_params)
            else:
                freq_params_json = frequency_params
        
        # Convert adjacency matrix to binary if it exists
        adj_matrix_binary = None
        if adjacency_matrix is not None:
            adj_matrix_binary = pickle.dumps(adjacency_matrix)
        
        # Create the configuration record
        config = Configuration(
            name=name,
            n_oscillators=n_oscillators,
            coupling_strength=coupling_strength,
            simulation_time=simulation_time,
            time_step=time_step,
            random_seed=random_seed,
            network_type=network_type,
            frequency_distribution=frequency_distribution,
            frequency_params=freq_params_json,
            adjacency_matrix=adj_matrix_binary
        )
        
        # Add to session and commit
        session.add(config)
        session.commit()
        
        return config.id
    
    except Exception as e:
        session.rollback()
        print(f"Error saving configuration: {e}")
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
    # Create a session
    session = Session()
    
    try:
        # Query all configurations
        configs = session.query(Configuration).order_by(Configuration.timestamp.desc()).all()
        
        # Create a list of result dictionaries
        results = []
        for config in configs:
            # Parse frequency params if available
            freq_params = None
            if config.frequency_params:
                try:
                    freq_params = json.loads(config.frequency_params)
                except:
                    freq_params = config.frequency_params
            
            # Add result to the list
            results.append({
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
        
        return results
    
    except Exception as e:
        print(f"Error listing configurations: {e}")
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
    # Create a session
    session = Session()
    
    try:
        # Query the configuration
        config = session.query(Configuration).get(config_id)
        
        if not config:
            return None
        
        # Parse frequency params if available
        freq_params = None
        if config.frequency_params:
            try:
                freq_params = json.loads(config.frequency_params)
            except:
                freq_params = config.frequency_params
        
        # Get adjacency matrix if available
        adj_matrix = None
        if config.adjacency_matrix:
            adj_matrix = pickle.loads(config.adjacency_matrix)
        
        # Create a result dictionary
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
        print(f"Error retrieving configuration: {e}")
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
    # Create a session
    session = Session()
    
    try:
        # Query the configuration by name
        config = session.query(Configuration).filter_by(name=name).first()
        
        if not config:
            return None
        
        # Parse frequency params if available
        freq_params = None
        if config.frequency_params:
            try:
                freq_params = json.loads(config.frequency_params)
            except:
                freq_params = config.frequency_params
        
        # Get adjacency matrix if available
        adj_matrix = None
        if config.adjacency_matrix:
            adj_matrix = pickle.loads(config.adjacency_matrix)
        
        # Create a result dictionary
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
        print(f"Error retrieving configuration by name: {e}")
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
    # Create a session
    session = Session()
    
    try:
        # Query the configuration
        config = session.query(Configuration).get(config_id)
        
        if not config:
            return False
        
        # Delete the configuration
        session.delete(config)
        session.commit()
        
        return True
    
    except Exception as e:
        session.rollback()
        print(f"Error deleting configuration: {e}")
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
    # Get the configuration
    config = get_configuration(config_id)
    
    if not config:
        print(f"Configuration with ID {config_id} not found")
        return None
    
    # Create a serializable dictionary
    export_dict = {
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
    
    # Add adjacency matrix if available
    if config['adjacency_matrix'] is not None:
        export_dict['adjacency_matrix'] = config['adjacency_matrix'].tolist()
    
    # Convert to JSON
    json_str = json.dumps(export_dict, indent=2)
    
    # Save to file if path is provided
    if file_path:
        try:
            with open(file_path, 'w') as f:
                f.write(json_str)
            print(f"Configuration exported to {file_path}")
            return None
        except Exception as e:
            print(f"Error exporting configuration: {e}")
            return None
    
    # Return JSON string if no file path
    return json_str


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
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        
        # Convert lists back to numpy arrays if needed
        if config_dict.get('adjacency_matrix') is not None:
            try:
                # Ensure adjacency matrix is a numpy array
                adj_matrix = np.array(config_dict['adjacency_matrix'])
                
                # Make sure no self-loops (diagonal should be zero)
                np.fill_diagonal(adj_matrix, 0)
                
                print(f"Converted adjacency matrix to numpy array:")
                print(f"- Shape: {adj_matrix.shape}")
                print(f"- Sum: {np.sum(adj_matrix)}")
                print(f"- Non-zero elements: {np.count_nonzero(adj_matrix)}")
                
                # Determine if it's a fully connected matrix (all 1s except diagonal)
                is_fully_connected = np.all(
                    (adj_matrix == 1) | 
                    (np.eye(adj_matrix.shape[0]) == 1)
                )
                
                if is_fully_connected:
                    print("WARNING: Matrix appears to be fully connected. Checking if this is correct...")
                    # If every off-diagonal element is 1, it might be mistakenly set to fully connected
                    if np.sum(adj_matrix) == adj_matrix.shape[0] * (adj_matrix.shape[0] - 1):
                        print("This seems to be a 'All-to-All' network type rather than a custom matrix.")
                
                config_dict['adjacency_matrix'] = adj_matrix
                
                # Verify network type is set correctly
                if config_dict.get('network_type') != "Custom Adjacency Matrix":
                    print(f"Updating network type from {config_dict.get('network_type')} to 'Custom Adjacency Matrix'")
                    config_dict['network_type'] = "Custom Adjacency Matrix"
            except Exception as e:
                print(f"Error processing adjacency matrix: {e}")
                config_dict['adjacency_matrix'] = None
        
        # Save to database if requested
        if save_to_db:
            # Generate a unique name if needed
            name = config_dict.get('name', os.path.basename(file_path).split('.')[0])
            session = Session()
            existing = session.query(Configuration).filter_by(name=name).first()
            session.close()
            
            if existing:
                name = f"{name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Save to database - prepare frequency_params
            freq_params = config_dict.get('frequency_params')
            if isinstance(freq_params, str):
                try:
                    freq_params = json.loads(freq_params)
                except:
                    pass  # Keep as string if JSON parsing fails
            
            # Handle adjacency matrix
            adj_matrix = config_dict.get('adjacency_matrix')
            
            # Call the save_configuration function
            config_id = save_configuration(
                name=name,
                n_oscillators=config_dict.get('n_oscillators'),
                coupling_strength=config_dict.get('coupling_strength'),
                simulation_time=config_dict.get('simulation_time'),
                time_step=config_dict.get('time_step'),
                random_seed=config_dict.get('random_seed'),
                network_type=config_dict.get('network_type'),
                frequency_distribution=config_dict.get('frequency_distribution'),
                frequency_params=freq_params,
                adjacency_matrix=adj_matrix
            )
            
            return config_id
        
        return config_dict
    
    except Exception as e:
        print(f"Error importing configuration: {e}")
        return None


# ML Dataset functions

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
    session = Session()
    
    try:
        # Create the dataset
        dataset = MLDataset(
            name=name,
            description=description,
            feature_type=feature_type,
            target_type=target_type
        )
        
        # Add to session and commit
        session.add(dataset)
        session.commit()
        
        return dataset.id
    
    except Exception as e:
        session.rollback()
        print(f"Error creating dataset: {e}")
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
    session = Session()
    
    try:
        # Check if the dataset and simulation exist
        dataset = session.query(MLDataset).get(dataset_id)
        simulation = session.query(Simulation).get(simulation_id)
        
        if not dataset or not simulation:
            return False
        
        # Check if the simulation is already in the dataset
        existing = session.query(MLDatasetSimulation).filter_by(
            dataset_id=dataset_id, 
            simulation_id=simulation_id
        ).first()
        
        if existing:
            # Update the split if it's different
            if existing.split != split:
                existing.split = split
                session.commit()
            return True
        
        # Add the simulation to the dataset
        dataset_sim = MLDatasetSimulation(
            dataset_id=dataset_id,
            simulation_id=simulation_id,
            split=split
        )
        
        # Add to session and commit
        session.add(dataset_sim)
        session.commit()
        
        return True
    
    except Exception as e:
        session.rollback()
        print(f"Error adding simulation to dataset: {e}")
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
    session = Session()
    
    try:
        # Get the dataset and its simulations
        dataset = session.query(MLDataset).get(dataset_id)
        if not dataset:
            return []
        
        dataset_sims = session.query(MLDatasetSimulation).filter_by(dataset_id=dataset_id).all()
        if not dataset_sims:
            return []
        
        # Load all the simulations
        simulations = []
        for ds in dataset_sims:
            sim_data = get_simulation(ds.simulation_id)
            if sim_data:
                sim_data['split'] = ds.split
                simulations.append(sim_data)
        
        # Extract features
        feature_ids = []
        for feature_name, config in feature_config.items():
            # Initialize storage for extracted data
            feature_data = {'value': [], 'simulation_ids': [], 'splits': []}
            
            # Extract feature from each simulation
            for sim in simulations:
                # Extract based on the specified method
                extraction_method = config['extraction']
                
                # Parse nested attributes (e.g., 'order_parameter.r')
                parts = extraction_method.split('.')
                value = sim
                for part in parts:
                    if part == 'params':
                        value = sim['params']
                    elif isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        break
                
                # Store the extracted feature
                feature_data['value'].append(value)
                feature_data['simulation_ids'].append(sim['params']['id'])
                feature_data['splits'].append(sim['split'])
            
            # Serialize and store the feature
            serialized_data = pickle.dumps(feature_data)
            
            # Create or update feature record
            existing_feature = session.query(MLFeature).filter_by(
                dataset_id=dataset_id,
                name=feature_name
            ).first()
            
            if existing_feature:
                existing_feature.feature_type = config['type']
                existing_feature.description = config.get('description', '')
                existing_feature.extraction_method = extraction_method
                existing_feature.data = serialized_data
                feature_ids.append(existing_feature.id)
            else:
                feature = MLFeature(
                    dataset_id=dataset_id,
                    name=feature_name,
                    feature_type=config['type'],
                    description=config.get('description', ''),
                    extraction_method=extraction_method,
                    data=serialized_data
                )
                session.add(feature)
                session.flush()
                feature_ids.append(feature.id)
        
        # Commit changes
        session.commit()
        return feature_ids
    
    except Exception as e:
        session.rollback()
        print(f"Error extracting features: {e}")
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
    session = Session()
    
    try:
        # Get the dataset
        dataset = session.query(MLDataset).get(dataset_id)
        if not dataset:
            return None
        
        # Get dataset simulations
        dataset_sims = session.query(MLDatasetSimulation).filter_by(dataset_id=dataset_id).all()
        simulations = []
        for ds in dataset_sims:
            sim = session.query(Simulation).get(ds.simulation_id)
            if sim:
                simulations.append({
                    'id': sim.id,
                    'n_oscillators': sim.n_oscillators,
                    'coupling_strength': sim.coupling_strength,
                    'split': ds.split
                })
        
        # Get dataset features
        features = []
        feature_records = session.query(MLFeature).filter_by(dataset_id=dataset_id).all()
        for feature in feature_records:
            # Deserialize the data
            feature_data = pickle.loads(feature.data)
            
            features.append({
                'id': feature.id,
                'name': feature.name,
                'type': feature.feature_type,
                'description': feature.description,
                'data': feature_data
            })
        
        # Create result dictionary
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
        print(f"Error getting dataset: {e}")
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
    session = Session()
    
    try:
        datasets = session.query(MLDataset).order_by(MLDataset.timestamp.desc()).all()
        
        results = []
        for dataset in datasets:
            # Count simulations in each split
            sim_counts = {}
            sim_splits = session.query(MLDatasetSimulation.split, 
                                      MLDatasetSimulation.simulation_id).filter_by(
                dataset_id=dataset.id).all()
            
            for split, _ in sim_splits:
                sim_counts[split] = sim_counts.get(split, 0) + 1
            
            # Count features by type
            feature_counts = {}
            feature_types = session.query(MLFeature.feature_type, 
                                         MLFeature.id).filter_by(
                dataset_id=dataset.id).all()
            
            for ftype, _ in feature_types:
                feature_counts[ftype] = feature_counts.get(ftype, 0) + 1
            
            results.append({
                'id': dataset.id,
                'name': dataset.name,
                'description': dataset.description,
                'timestamp': dataset.timestamp,
                'feature_type': dataset.feature_type,
                'target_type': dataset.target_type,
                'simulation_counts': sim_counts,
                'feature_counts': feature_counts
            })
        
        return results
    
    except Exception as e:
        print(f"Error listing datasets: {e}")
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
    # Get the dataset
    dataset = get_ml_dataset(dataset_id)
    if not dataset:
        print(f"Dataset with ID {dataset_id} not found")
        return None
    
    # Create export directory
    if export_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        export_dir = f"ml_dataset_{dataset['name']}_{timestamp}"
    
    os.makedirs(export_dir, exist_ok=True)
    
    # Check if dataset has predefined splits
    has_splits = any(sim.get('split') for sim in dataset['simulations'])
    
    # Create subdirectories for splits
    if has_splits:
        os.makedirs(os.path.join(export_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(export_dir, 'validation'), exist_ok=True)
        os.makedirs(os.path.join(export_dir, 'test'), exist_ok=True)
    else:
        # Create random splits
        n_samples = len(dataset['simulations'])
        indices = np.random.permutation(n_samples)
        
        train_end = int(train_ratio * n_samples)
        val_end = train_end + int(val_ratio * n_samples)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        # Assign splits
        for i, sim in enumerate(dataset['simulations']):
            if i in train_indices:
                sim['split'] = 'train'
            elif i in val_indices:
                sim['split'] = 'validation'
            else:
                sim['split'] = 'test'
        
        # Create directories
        os.makedirs(os.path.join(export_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(export_dir, 'validation'), exist_ok=True)
        os.makedirs(os.path.join(export_dir, 'test'), exist_ok=True)
    
    # Export metadata
    metadata = {
        'name': dataset['name'],
        'description': dataset['description'],
        'feature_type': dataset['feature_type'],
        'target_type': dataset['target_type'],
        'n_simulations': len(dataset['simulations']),
        'n_features': len(dataset['features']),
        'export_date': datetime.now().isoformat(),
        'export_format': format
    }
    
    with open(os.path.join(export_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Export features for each split
    for feature in dataset['features']:
        feature_name = feature['name']
        feature_data = feature['data']
        
        # Group by split
        train_data = []
        val_data = []
        test_data = []
        
        for i, split in enumerate(feature_data['splits']):
            if split == 'train':
                train_data.append(feature_data['value'][i])
            elif split == 'validation':
                val_data.append(feature_data['value'][i])
            else:  # test
                test_data.append(feature_data['value'][i])
        
        # Export based on format
        if format == 'numpy':
            for split_name, data in [('train', train_data), ('validation', val_data), ('test', test_data)]:
                if data:
                    # Convert to numpy array if possible
                    try:
                        np_data = np.array(data)
                        np.save(os.path.join(export_dir, split_name, f"{feature_name}.npy"), np_data)
                    except:
                        # Fallback to pickle if cannot convert to numpy
                        with open(os.path.join(export_dir, split_name, f"{feature_name}.pkl"), 'wb') as f:
                            pickle.dump(data, f)
        
        elif format == 'pandas':
            for split_name, data in [('train', train_data), ('validation', val_data), ('test', test_data)]:
                if data:
                    try:
                        df = pd.DataFrame(data)
                        df.to_csv(os.path.join(export_dir, split_name, f"{feature_name}.csv"))
                    except:
                        # Fallback to pickle
                        with open(os.path.join(export_dir, split_name, f"{feature_name}.pkl"), 'wb') as f:
                            pickle.dump(data, f)
        
        elif format == 'pickle':
            for split_name, data in [('train', train_data), ('validation', val_data), ('test', test_data)]:
                if data:
                    with open(os.path.join(export_dir, split_name, f"{feature_name}.pkl"), 'wb') as f:
                        pickle.dump(data, f)
    
    print(f"Dataset exported to {export_dir}")
    return export_dir


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
    from src.models.kuramoto_model import KuramotoModel
    
    # Create dataset if name is provided
    dataset_id = None
    if dataset_name:
        dataset_id = create_ml_dataset(
            name=dataset_name,
            description=f"Batch simulations with {len(config_variations)} variations",
            feature_type='time_series',
            target_type='regression'
        )
    
    # Run simulations
    simulation_results = []
    
    for i, variation in enumerate(config_variations):
        print(f"Running simulation {i+1}/{len(config_variations)}")
        
        # Create config by combining base_config and variation
        config = base_config.copy()
        config.update(variation)
        
        # Create model
        n_oscillators = config.get('n_oscillators', 10)
        coupling_strength = config.get('coupling_strength', 1.0)
        simulation_time = config.get('simulation_time', 10.0)
        time_step = config.get('time_step', 0.01)
        random_seed = config.get('random_seed', None)
        
        # Handle frequency distribution
        frequencies = None
        freq_type = config.get('frequency_distribution', 'normal')
        freq_params = config.get('frequency_params', {'mean': 0.0, 'std': 0.1})
        
        if freq_type == 'normal':
            np.random.seed(random_seed)
            frequencies = np.random.normal(
                freq_params.get('mean', 0.0),
                freq_params.get('std', 0.1),
                n_oscillators
            )
        elif freq_type == 'uniform':
            np.random.seed(random_seed)
            frequencies = np.random.uniform(
                freq_params.get('low', -0.5),
                freq_params.get('high', 0.5),
                n_oscillators
            )
        elif freq_type == 'custom' and 'values' in freq_params:
            frequencies = np.array(freq_params['values'])
        
        # Handle network type
        adjacency_matrix = None
        network_type = config.get('network_type', 'all-to-all')
        
        if network_type == 'ring':
            # Create ring network (each oscillator connected to neighbors)
            adjacency_matrix = np.zeros((n_oscillators, n_oscillators))
            for i in range(n_oscillators):
                adjacency_matrix[i, (i + 1) % n_oscillators] = 1
                adjacency_matrix[i, (i - 1) % n_oscillators] = 1
        elif network_type == 'random':
            # Create random network with given density
            np.random.seed(random_seed)
            density = config.get('network_density', 0.3)
            adjacency_matrix = np.random.random((n_oscillators, n_oscillators)) < density
            np.fill_diagonal(adjacency_matrix, 0)
            # Ensure symmetry
            adjacency_matrix = np.logical_or(adjacency_matrix, adjacency_matrix.T).astype(int)
        elif network_type == 'custom' and 'matrix' in config:
            adjacency_matrix = np.array(config['matrix'])
        
        # Create and run model
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
        
        # Store simulation
        sim_id = store_simulation(
            model=model,
            times=times,
            phases=phases,
            order_parameter=order_parameter,
            frequencies=frequencies,
            freq_type=freq_type,
            freq_params=freq_params,
            adjacency_matrix=adjacency_matrix
        )
        
        # Add to dataset if created
        if dataset_id:
            # Determine split (80% train, 10% validation, 10% test by default)
            split_ratio = i / len(config_variations)
            if split_ratio < 0.8:
                split = 'train'
            elif split_ratio < 0.9:
                split = 'validation'
            else:
                split = 'test'
            
            add_simulation_to_dataset(dataset_id, sim_id, split=split)
        
        # Record result
        simulation_results.append({
            'id': sim_id,
            'config': config,
            'split': split
        })
    
    # Return the batch results
    return {
        'dataset_id': dataset_id,
        'simulations': simulation_results,
        'total_simulations': len(simulation_results)
    }