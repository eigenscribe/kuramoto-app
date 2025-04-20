"""
Database module for the Kuramoto Model Simulator.
This module provides functionality to store and retrieve simulation data.
"""

import os
import numpy as np
import json
import pickle
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
    
    frequencies = relationship("Frequency", back_populates="simulation", cascade="all, delete-orphan")
    phases = relationship("Phase", back_populates="simulation", cascade="all, delete-orphan")
    order_parameters = relationship("OrderParameter", back_populates="simulation", cascade="all, delete-orphan")
    adjacency_matrix = relationship("AdjacencyMatrix", back_populates="simulation", uselist=False, cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Simulation(id={self.id}, n_oscillators={self.n_oscillators}, coupling={self.coupling_strength})>"


class Frequency(Base):
    """Model representing oscillator natural frequencies."""
    __tablename__ = "frequencies"
    
    id = Column(Integer, primary_key=True)
    simulation_id = Column(Integer, ForeignKey("simulations.id"))
    oscillator_index = Column(Integer)
    value = Column(Float)
    
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
    
    simulation = relationship("Simulation", back_populates="order_parameters")
    
    def __repr__(self):
        return f"<OrderParameter(time={self.time_index}, magnitude={self.magnitude})>"


class AdjacencyMatrix(Base):
    """Model representing the network adjacency matrix."""
    __tablename__ = "adjacency_matrices"
    
    id = Column(Integer, primary_key=True)
    simulation_id = Column(Integer, ForeignKey("simulations.id"), unique=True)
    data = Column(LargeBinary)  # Stored as a pickled numpy array
    
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

# Create session
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
    
    try:
        # Create simulation record
        sim = Simulation(
            n_oscillators=model.n_oscillators,
            coupling_strength=model.coupling_strength,
            simulation_time=model.simulation_time,
            time_step=model.time_step,
            random_seed=model.random_seed if hasattr(model, 'random_seed') else None,
            frequency_distribution=freq_type,
            frequency_params=json.dumps(freq_params) if freq_params else None
        )
        session.add(sim)
        session.flush()  # Flush to get sim.id
        
        # Store frequencies
        for i, freq in enumerate(frequencies):
            session.add(Frequency(
                simulation_id=sim.id,
                oscillator_index=i,
                value=float(freq)
            ))
        
        # Store phases - only store 1 out of 10 time points to save space
        for t_idx, t in enumerate(times[::10]):
            for osc_idx, phase in enumerate(phases[:, t_idx*10]):
                session.add(Phase(
                    simulation_id=sim.id,
                    time_index=t_idx,
                    oscillator_index=osc_idx,
                    value=float(phase)
                ))
                
        # Store order parameter
        for t_idx, t in enumerate(times[::10]):
            r = order_parameter[t_idx*10]
            psi = np.angle(np.sum(np.exp(1j * phases[:, t_idx*10])))
            
            session.add(OrderParameter(
                simulation_id=sim.id,
                time_index=t_idx,
                magnitude=float(r),
                phase=float(psi)
            ))
            
        # Store adjacency matrix if provided
        if adjacency_matrix is not None:
            session.add(AdjacencyMatrix(
                simulation_id=sim.id,
                data=pickle.dumps(adjacency_matrix)
            ))
            
        session.commit()
        return sim.id
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


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
    
    try:
        sim = session.query(Simulation).filter_by(id=simulation_id).first()
        
        if not sim:
            return None
            
        # Get frequencies
        frequencies = [freq.value for freq in sim.frequencies]
        
        # Get phases
        phases_data = sim.phases
        n_oscillators = sim.n_oscillators
        time_indices = sorted(set(p.time_index for p in phases_data))
        
        # Reshape phases into a 2D array
        phases = np.zeros((n_oscillators, len(time_indices)))
        for p in phases_data:
            phases[p.oscillator_index, p.time_index] = p.value
            
        # Get order parameter
        order_param = np.array([op.magnitude for op in sim.order_parameters])
        
        # Get adjacency matrix if available
        adj_matrix = None
        if sim.adjacency_matrix:
            adj_matrix = pickle.loads(sim.adjacency_matrix.data)
            
        freq_params = json.loads(sim.frequency_params) if sim.frequency_params else None
            
        return {
            'id': sim.id,
            'timestamp': sim.timestamp,
            'n_oscillators': sim.n_oscillators,
            'coupling_strength': sim.coupling_strength,
            'simulation_time': sim.simulation_time,
            'time_step': sim.time_step,
            'random_seed': sim.random_seed,
            'frequency_distribution': sim.frequency_distribution,
            'frequency_params': freq_params,
            'frequencies': np.array(frequencies),
            'phases': phases,
            'order_parameter': order_param,
            'adjacency_matrix': adj_matrix
        }
    finally:
        session.close()


def list_simulations():
    """
    List all simulations in the database.
    
    Returns:
    --------
    list
        List of dictionaries containing basic simulation info
    """
    session = Session()
    
    try:
        sims = session.query(Simulation).order_by(Simulation.timestamp.desc()).all()
        return [{
            'id': sim.id,
            'timestamp': sim.timestamp,
            'n_oscillators': sim.n_oscillators,
            'coupling_strength': sim.coupling_strength,
            'simulation_time': sim.simulation_time,
            'frequency_distribution': sim.frequency_distribution
        } for sim in sims]
    finally:
        session.close()


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
    
    try:
        sim = session.query(Simulation).filter_by(id=simulation_id).first()
        
        if not sim:
            return False
            
        session.delete(sim)
        session.commit()
        return True
    except Exception:
        session.rollback()
        return False
    finally:
        session.close()


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
    
    try:
        # Convert adjacency matrix to binary if provided
        binary_adj_matrix = None
        if adjacency_matrix is not None:
            binary_adj_matrix = pickle.dumps(adjacency_matrix)
        
        # Convert frequency params to JSON
        params_json = json.dumps(frequency_params) if frequency_params else None
        
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
            existing.frequency_params = params_json
            existing.adjacency_matrix = binary_adj_matrix
            existing.timestamp = datetime.now()
            session.commit()
            return existing.id
        else:
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
                frequency_params=params_json,
                adjacency_matrix=binary_adj_matrix
            )
            session.add(config)
            session.commit()
            return config.id
    except Exception as e:
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
        List of dictionaries containing configuration info
    """
    session = Session()
    
    try:
        configs = session.query(Configuration).order_by(Configuration.name).all()
        return [{
            'id': config.id,
            'name': config.name,
            'timestamp': config.timestamp,
            'n_oscillators': config.n_oscillators,
            'coupling_strength': config.coupling_strength,
            'network_type': config.network_type,
            'frequency_distribution': config.frequency_distribution
        } for config in configs]
    finally:
        session.close()


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
    
    try:
        config = session.query(Configuration).filter_by(id=config_id).first()
        
        if not config:
            return None
            
        # Parse JSON and binary data
        freq_params = json.loads(config.frequency_params) if config.frequency_params else None
        adj_matrix = pickle.loads(config.adjacency_matrix) if config.adjacency_matrix else None
            
        return {
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
    finally:
        session.close()


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
    
    try:
        config = session.query(Configuration).filter_by(id=config_id).first()
        
        if not config:
            return False
            
        session.delete(config)
        session.commit()
        return True
    except Exception:
        session.rollback()
        return False
    finally:
        session.close()


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
    
    try:
        config = session.query(Configuration).filter_by(name=name).first()
        
        if not config:
            return None
            
        # Parse JSON and binary data
        freq_params = json.loads(config.frequency_params) if config.frequency_params else None
        adj_matrix = pickle.loads(config.adjacency_matrix) if config.adjacency_matrix else None
            
        return {
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
    finally:
        session.close()