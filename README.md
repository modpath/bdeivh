# BDEIVH

The Birth-Death Exposed-Infectious Vector-Human (BDEIVH) model is a special case of the Multi-Type Birth-Death model [[Stadler _et al._ 2013]](https://royalsocietypublishing.org/doi/10.1098/rstb.2012.0198).

The BDEIVH model has 4 states:
* V<sub>E</sub>, exposed vector, i.e. infected but not yet infectious
* V<sub>I</sub>, infectious vector
* H<sub>E</sub>, exposed human, i.e. infected but not yet infectious
* H<sub>I</sub>, infectious human

It has 2 non-zero state transition rate parameters:
* μ<sub>V</sub> -- transition rate from state vi to state ve, corresponding to the incubation in vectors
* μ<sub>H</sub> -- transition rate from state hi to state he, corresponding to the incubation in humans

It has 2 non-zero transmission rate parameters:
* λ<sub>V</sub> -- transmission rate from state vi (donor) to state he (recipient), corresponding to the vector-to-human transmission
* λ<sub>H</sub> -- transmission rate from state hi (donor) to state ve (recipient), corresponding to the human-to-vector transmission

It also has 2 non-zero removal (becoming non-infectious) rate parameters:
* ψ<sub>V</sub> -- removal rate of state vi, corresponding to the death (or recovery) of infectious vectors
* ψ<sub>H</sub> -- removal rate of state hi, corresponding to the recovery, self-isolation (or death) of infectious humans

Finally, it has one non-zero sampling probability upon removal parameter:
* ρ<sub>H</sub> -- probability to sample the pathogen of a human upon removal

Using these probabilities, one can calculate the equilibrium frequencies of the model's states, 
where 0 ≤ π<sub>j</sub> ≤ 1, Σ<sub>j</sub>π<sub>j</sub> = 1, and j ∈ {V<sub>E</sub>, V<sub>I</sub>, H<sub>E</sub>, H<sub>I</sub>}. 
We can also define the equilibrium frequency of humans and vectors as:
* π<sub>H</sub>=π<sub>H<sub>E</sub></sub> + π<sub>H<sub>I</sub></sub>, and
* π<sub>V</sub>=π<sub>V<sub>E</sub></sub> + π<sub>V<sub>I</sub></sub> respectively.

The BDEIVH model has the following epidemiological parameters:

* R<sub>HV</sub> = λ<sub>H</sub>/ψ<sub>H</sub> -- human-to-vector reproduction number
* R<sub>VH</sub> = λ<sub>V</sub>/ψ<sub>V</sub> -- vector-to-human reproduction number
* d<sub>inc,V</sub> = 1/μ<sub>V</sub> -- vector incubation period
* d<sub>inc,H</sub> = 1/μ<sub>H</sub> -- human incubation period
* d<sub>inf,V</sub> = 1/ψ<sub>V</sub> -- vector infectious time
* d<sub>inf,H</sub> = 1/ψ<sub>H</sub> -- human infectious time
* d<sub>V</sub> = d<sub>inc,V</sub> + d<sub>inf,V</sub> -- total vector infection time
* d<sub>H</sub> = d<sub>inc,H</sub> + d<sub>inf,H</sub> -- total human infection time
* R = π<sub>H</sub>R<sub>H</sub> + π<sub>V</sub>R<sub>V</sub> -- host-unaware reproduction number
* d = π<sub>H</sub>d<sub>H</sub> + π<sub>V</sub>d<sub>V</sub> -- host-unaware infection time
* R<sub>HH</sub> = R<sub>HV</sub>R<sub>VH</sub> -- human-to-human reproduction number
* π<sub>V</sub> / π<sub>H</sub> -- vector to human ratio