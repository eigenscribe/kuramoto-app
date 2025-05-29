"""
Database module for the Kuramoto Model Simulator.
This module provides functionality to store and retrieve simulation data,
as well as import/export configurations as JSON files.
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
        
        result = []
        for sim in simulations:
            # Get the network type
            adj_matrix = session.query(AdjacencyMatrix).filter_by(simulation_id=sim.id).first()
            network_type = adj_matrix.network_type if adj_matrix else "All-to-All"
            
            # Parse frequency params if available
            freq_params = None
            if sim.frequency_params:
                try:
                    freq_params = json.loads(sim.frequency_params)
                except:
                    freq_params = sim.frequency_params
            
            # Add to result
            result.append({
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
        
        return result
    
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
        
        # Delete the simulation (cascade will delete related records)
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
                     random_seed, network_type, frequency_distribution, frequency_params, adjacency_matrix=None):
    """
    Save a configuration to the database.
    
    Parameters:
    -----------
    name : str
        Name for the configuration
    n_oscillators : int
        Number of oscillators
    coupling_strength : float
        Coupling strength K
    simulation_time : float
        Total simulation time
    time_step : float
        Time step for simulation
    random_seed : int
        Random seed for reproducibility
    network_type : str
        Type of network connection ('All-to-All', 'Nearest Neighbor', 'Random', 'Custom Adjacency Matrix')
    frequency_distribution : str
        Type of frequency distribution ('Normal', 'Uniform', 'Bimodal', 'Custom')
    frequency_params : str or dict
        Parameters for the frequency distribution (will be converted to JSON if dict)
    adjacency_matrix : ndarray, optional
        Custom adjacency matrix (for 'Custom Adjacency Matrix' network type)
        
    Returns:
    --------
    int
        ID of the saved configuration
    """
    # Create a session
    session = Session()
    
    try:
        # Convert adjacency matrix to binary if it exists
        adj_matrix_binary = None
        if adjacency_matrix is not None:
            adj_matrix_binary = pickle.dumps(adjacency_matrix)
        
        # Convert frequency params to JSON if it's a dict
        if isinstance(frequency_params, dict):
            frequency_params = json.dumps(frequency_params)
        
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
            frequency_params=frequency_params,
            adjacency_matrix=adj_matrix_binary
        )
        
        # Add to session and commit
        session.add(config)
        session.commit()
        
        return config.id
    
    except Exception as e:
        session.rollback()
        print(f"Error saving configuration: {e}")
        raise
    
    finally:
        session.close()


def list_configurations():
    """
    List all configurations in the database.
    
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
        
        result = []
        for config in configs:
            # Parse frequency params if available
            freq_params = None
            if config.frequency_params:
                try:
                    freq_params = json.loads(config.frequency_params)
                except:
                    freq_params = config.frequency_params
            
            # Add to result
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
                'has_matrix': config.adjacency_matrix is not None
            })
        
        return result
    
    except Exception as e:
        print(f"Error listing configurations: {e}")
        return []
    
    finally:
        session.close()


def get_configuration(config_id):
    """
    Get a configuration from the database by ID.
    
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
        
        # Create result dictionary
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
    Get a configuration from the database by name.
    
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
        # Query the configuration
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
        
        # Create result dictionary
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
    Export a configuration to a JSON file.
    
    Parameters:
    -----------
    config_id : int
        ID of the configuration to export
    file_path : str, optional
        Path where to save the JSON file, if None, returns the JSON string
        
    Returns:
    --------
    str or None
        JSON string if file_path is None, otherwise None
    """
    # Get the configuration
    config = get_configuration(config_id)
    
    if not config:
        return None
    
    # Create a JSON-compatible dictionary
    json_data = {
        'n_oscillators': config['n_oscillators'],
        'coupling_strength': config['coupling_strength'],
        'network_type': config['network_type'],
        'simulation_time': config['simulation_time'],
        'time_step': config['time_step'],
        'random_seed': config['random_seed'],
        'frequency_distribution': config['frequency_distribution'],
        'frequency_parameters': config['frequency_params']
    }
    
    # Add adjacency matrix if available
    if config['adjacency_matrix'] is not None:
        json_data['adjacency_matrix'] = config['adjacency_matrix'].tolist()
    
    # Export to file or return as string
    if file_path:
        with open(file_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        return None
    else:
        return json.dumps(json_data, indent=2)


def import_configuration_from_json(file_path, save_to_db=True):
    """
    Import a configuration from a JSON file.
    
    Parameters:
    -----------
    file_path : str
        Path to the JSON file
    save_to_db : bool, optional
        Whether to save the imported configuration to the database
        
    Returns:
    --------
    dict
        Dictionary containing the imported configuration
    """
    try:
        # Read the JSON file
        with open(file_path, 'r') as f:
            json_data = json.load(f)
        
        # Process adjacency matrix if available
        adj_matrix = None
        if 'adjacency_matrix' in json_data:
            adj_matrix = np.array(json_data['adjacency_matrix'])
        
        # Create a configuration dictionary
        config = {
            'n_oscillators': json_data.get('n_oscillators', 10),
            'coupling_strength': json_data.get('coupling_strength', 1.0),
            'network_type': json_data.get('network_type', 'All-to-All'),
            'simulation_time': json_data.get('simulation_time', 10.0),
            'time_step': json_data.get('time_step', 0.01),
            'random_seed': json_data.get('random_seed', 42),
            'frequency_distribution': json_data.get('frequency_distribution', 'Normal'),
            'frequency_params': json_data.get('frequency_parameters', {'mean': 0.0, 'std': 0.2}),
            'adjacency_matrix': adj_matrix
        }
        
        # Save to database if requested
        if save_to_db:
            name = os.path.splitext(os.path.basename(file_path))[0]
            config_id = save_configuration(
                name=name,
                n_oscillators=config['n_oscillators'],
                coupling_strength=config['coupling_strength'],
                simulation_time=config['simulation_time'],
                time_step=config['time_step'],
                random_seed=config['random_seed'],
                network_type=config['network_type'],
                frequency_distribution=config['frequency_distribution'],
                frequency_params=config['frequency_params'],
                adjacency_matrix=config['adjacency_matrix']
            )
            config['id'] = config_id
        
        return config
    
    except Exception as e:
        print(f"Error importing configuration: {e}")
        return None