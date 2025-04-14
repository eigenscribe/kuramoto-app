"""
Database module for the Kuramoto Model Simulator.
This module provides functionality to store and retrieve simulation data.
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