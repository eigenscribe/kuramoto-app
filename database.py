"""
Database module for the Kuramoto Model Simulator.
This module provides functionality to store and retrieve simulation data,
as well as import/export configurations as JSON files.
"""

import os
import numpy as np
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, LargeBinary, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Create the engine and base
DATABASE_URL = "sqlite:///kuramoto_simulations.db"
engine = create_engine(DATABASE_URL)
Base = declarative_base()

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
    data = Column(LargeBinary)  # Stored as a pickled numpy array
    
    # Relationship back to simulation
    simulation = relationship("Simulation", back_populates="adjacency_matrix")

    def __repr__(self):
        return f"<AdjacencyMatrix(simulation_id={self.simulation_id})>"

class Configuration(Base):
    """Model representing a saved simulation configuration."""
    __tablename__ = "configurations"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    timestamp = Column(DateTime, default=datetime.now)
    n_oscillators = Column(Integer)
    coupling_strength = Column(Float)
    simulation_time = Column(Float)
    time_step = Column(Float)
    random_seed = Column(Integer)
    network_type = Column(String(50))
    frequency_distribution = Column(String(50))
    frequency_params = Column(Text)  # JSON string of distribution parameters
    adjacency_matrix = Column(LargeBinary, nullable=True)  # For custom adjacency matrices
    
    def __repr__(self):
        return f"<Configuration(id={self.id}, name='{self.name}')>"

# Create all tables
Base.metadata.create_all(engine)

# Create a session factory
Session = sessionmaker(bind=engine)

def store_simulation(model, times, phases, order_parameter, frequencies, freq_type, freq_params=None, adjacency_matrix=None):
    """
    Store simulation data in the database.
    
    Parameters:
    -----------
    model : KuramotoModel
        The model object containing simulation parameters
    times : ndarray
        Time points of the simulation
    phases : ndarray
        Phases of oscillators at each time point
    order_parameter : ndarray
        Order parameter r(t) at each time point
    frequencies : ndarray
        Natural frequencies of the oscillators
    freq_type : str
        Type of frequency distribution (e.g., 'Normal', 'Uniform')
    freq_params : dict, optional
        Parameters of the frequency distribution
    adjacency_matrix : ndarray, optional
        Adjacency matrix representing network structure
        
    Returns:
    --------
    int
        The ID of the stored simulation
    """
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
    session.flush()  # Get the simulation ID without committing
    
    # Store natural frequencies
    for i, freq in enumerate(frequencies):
        session.add(Frequency(
            simulation_id=sim.id,
            oscillator_index=i,
            value=float(freq)
        ))
    
    # Store phases at selected time points (we can't store all time points for large simulations)
    # Take ~100 time points or all if there are fewer
    time_indices = np.linspace(0, len(times)-1, min(100, len(times)), dtype=int)
    for t_idx in time_indices:
        for i in range(model.n_oscillators):
            session.add(Phase(
                simulation_id=sim.id,
                time_index=t_idx,
                oscillator_index=i,
                value=float(phases[i, t_idx])
            ))
    
    # Store order parameter
    psi = np.angle(np.sum(np.exp(1j * phases), axis=0))
    for t_idx in time_indices:
        session.add(OrderParameter(
            simulation_id=sim.id,
            time_index=t_idx,
            magnitude=float(order_parameter[t_idx]),
            phase=float(psi[t_idx])
        ))
    
    # Store adjacency matrix if provided
    if adjacency_matrix is not None:
        session.add(AdjacencyMatrix(
            simulation_id=sim.id,
            data=adjacency_matrix.tobytes()
        ))
    
    session.commit()
    sim_id = sim.id
    session.close()
    
    return sim_id

def get_simulation(simulation_id):
    """
    Retrieve a simulation from the database.
    
    Parameters:
    -----------
    simulation_id : int
        The ID of the simulation to retrieve
        
    Returns:
    --------
    dict
        Dictionary containing simulation data
    """
    session = Session()
    
    sim = session.query(Simulation).filter_by(id=simulation_id).first()
    if not sim:
        session.close()
        return None
    
    # Get frequencies
    freqs = session.query(Frequency).filter_by(simulation_id=simulation_id).all()
    frequencies = np.array([f.value for f in sorted(freqs, key=lambda x: x.oscillator_index)])
    
    # Get phases
    phases_data = session.query(Phase).filter_by(simulation_id=simulation_id).all()
    time_indices = sorted(list(set([p.time_index for p in phases_data])))
    n_oscillators = sim.n_oscillators
    
    phases = np.zeros((n_oscillators, len(time_indices)))
    for phase in phases_data:
        t_idx = time_indices.index(phase.time_index)
        phases[phase.oscillator_index, t_idx] = phase.value
    
    # Get order parameters
    order_params = session.query(OrderParameter).filter_by(simulation_id=simulation_id).all()
    order_params.sort(key=lambda x: x.time_index)
    
    r = np.array([op.magnitude for op in order_params])
    psi = np.array([op.phase for op in order_params])
    
    # Get adjacency matrix if it exists
    adj_matrix = session.query(AdjacencyMatrix).filter_by(simulation_id=simulation_id).first()
    adjacency_matrix = None
    if adj_matrix:
        adjacency_matrix = np.frombuffer(adj_matrix.data).reshape((n_oscillators, n_oscillators))
    
    # Create the result dictionary
    result = {
        'id': sim.id,
        'timestamp': sim.timestamp,
        'n_oscillators': sim.n_oscillators,
        'coupling_strength': sim.coupling_strength,
        'simulation_time': sim.simulation_time,
        'time_step': sim.time_step,
        'random_seed': sim.random_seed,
        'frequency_distribution': sim.frequency_distribution,
        'frequency_params': json.loads(sim.frequency_params) if sim.frequency_params else None,
        'frequencies': frequencies,
        'phases': phases,
        'time_indices': time_indices,
        'order_parameter': {
            'r': r,
            'psi': psi
        },
        'adjacency_matrix': adjacency_matrix
    }
    
    session.close()
    return result

def list_simulations():
    """
    List all simulations in the database.
    
    Returns:
    --------
    list
        List of dictionaries containing basic simulation info
    """
    session = Session()
    
    sims = session.query(Simulation).all()
    result = [{
        'id': sim.id,
        'timestamp': sim.timestamp,
        'n_oscillators': sim.n_oscillators,
        'coupling_strength': sim.coupling_strength,
        'frequency_distribution': sim.frequency_distribution
    } for sim in sims]
    
    session.close()
    return result

def delete_simulation(simulation_id):
    """
    Delete a simulation from the database.
    
    Parameters:
    -----------
    simulation_id : int
        The ID of the simulation to delete
        
    Returns:
    --------
    bool
        True if deletion was successful, False otherwise
    """
    session = Session()
    
    sim = session.query(Simulation).filter_by(id=simulation_id).first()
    if not sim:
        session.close()
        return False
    
    session.delete(sim)
    session.commit()
    session.close()
    
    return True

def save_configuration(name, n_oscillators, coupling_strength, simulation_time, time_step, 
                      random_seed, network_type, frequency_distribution, frequency_params,
                      adjacency_matrix=None):
    """
    Save a simulation configuration to the database.
    
    Parameters:
    -----------
    name : str
        Name to identify this configuration
    n_oscillators : int
        Number of oscillators
    coupling_strength : float
        Coupling strength
    simulation_time : float
        Total simulation time
    time_step : float
        Simulation time step
    random_seed : int
        Random seed for reproducibility
    network_type : str
        Type of network connectivity
    frequency_distribution : str
        Type of frequency distribution
    frequency_params : dict
        Parameters of the frequency distribution
    adjacency_matrix : ndarray, optional
        Custom adjacency matrix for network connectivity
        
    Returns:
    --------
    int
        The ID of the saved configuration, or None if there was an error
    """
    session = Session()
    
    # Check if a configuration with this name already exists
    existing = session.query(Configuration).filter_by(name=name).first()
    if existing:
        session.close()
        return None
    
    # Prepare the adjacency matrix data if provided
    adj_matrix_data = None
    if adjacency_matrix is not None:
        adj_matrix_data = adjacency_matrix.tobytes()
    
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
        adjacency_matrix=adj_matrix_data
    )
    
    try:
        session.add(config)
        session.commit()
        config_id = config.id
        session.close()
        return config_id
    except Exception as e:
        session.rollback()
        session.close()
        raise e

def list_configurations():
    """
    List all saved configurations.
    
    Returns:
    --------
    list
        List of dictionaries containing configuration info
    """
    session = Session()
    
    configs = session.query(Configuration).all()
    result = [{
        'id': config.id,
        'name': config.name,
        'timestamp': config.timestamp,
        'n_oscillators': config.n_oscillators,
        'coupling_strength': config.coupling_strength,
        'network_type': config.network_type,
        'frequency_distribution': config.frequency_distribution
    } for config in configs]
    
    session.close()
    return result

def get_configuration(config_id):
    """
    Retrieve a configuration from the database.
    
    Parameters:
    -----------
    config_id : int
        The ID of the configuration to retrieve
        
    Returns:
    --------
    dict
        Dictionary containing configuration data
    """
    session = Session()
    
    config = session.query(Configuration).filter_by(id=config_id).first()
    if not config:
        session.close()
        return None
    
    # Process adjacency matrix if it exists
    adjacency_matrix = None
    if config.adjacency_matrix:
        try:
            adjacency_matrix = np.frombuffer(config.adjacency_matrix).reshape((config.n_oscillators, config.n_oscillators))
        except:
            # If there's an error reshaping, just leave as None
            pass
    
    # Create the result dictionary
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
        'frequency_params': json.loads(config.frequency_params) if config.frequency_params else None,
        'adjacency_matrix': adjacency_matrix
    }
    
    session.close()
    return result

def delete_configuration(config_id):
    """
    Delete a configuration from the database.
    
    Parameters:
    -----------
    config_id : int
        The ID of the configuration to delete
        
    Returns:
    --------
    bool
        True if deletion was successful, False otherwise
    """
    session = Session()
    
    config = session.query(Configuration).filter_by(id=config_id).first()
    if not config:
        session.close()
        return False
    
    session.delete(config)
    session.commit()
    session.close()
    
    return True

def get_configuration_by_name(name):
    """
    Retrieve a configuration by name.
    
    Parameters:
    -----------
    name : str
        The name of the configuration to retrieve
        
    Returns:
    --------
    dict
        Dictionary containing configuration data
    """
    session = Session()
    
    config = session.query(Configuration).filter_by(name=name).first()
    if not config:
        session.close()
        return None
    
    # Process adjacency matrix if it exists
    adjacency_matrix = None
    if config.adjacency_matrix:
        try:
            adjacency_matrix = np.frombuffer(config.adjacency_matrix).reshape((config.n_oscillators, config.n_oscillators))
        except:
            # If there's an error reshaping, just leave as None
            pass
    
    # Create the result dictionary
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
        'frequency_params': json.loads(config.frequency_params) if config.frequency_params else None,
        'adjacency_matrix': adjacency_matrix
    }
    
    session.close()
    return result
def export_configuration_to_json(config_id, file_path=None):
    """
    Export a saved configuration to a JSON file.
    
    Parameters:
    -----------
    config_id : int
        The ID of the configuration to export
    file_path : str, optional
        Path where the JSON file should be saved. If None, will use the config name.
        
    Returns:
    --------
    str
        Path to the exported JSON file or None if export failed
    """
    # Get the configuration
    config_dict = get_configuration(config_id)
    if not config_dict:
        return None
    
    # Determine file path if not provided
    if file_path is None:
        file_path = f"kuramoto_config_{config_dict['name'].replace(' ', '_')}.json"
    
    # Convert numpy arrays to lists for JSON serialization
    if config_dict.get('adjacency_matrix') is not None:
        config_dict['adjacency_matrix'] = config_dict['adjacency_matrix'].tolist()
    
    # Add metadata
    config_dict['export_date'] = datetime.now().isoformat()
    config_dict['version'] = '1.0'
    
    # Save to file
    try:
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        return file_path
    except Exception as e:
        print(f"Error exporting configuration: {e}")
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
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        
        # Convert lists back to numpy arrays if needed
        if config_dict.get('adjacency_matrix') is not None:
            config_dict['adjacency_matrix'] = np.array(config_dict['adjacency_matrix'])
        
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

