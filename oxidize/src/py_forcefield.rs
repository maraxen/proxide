use pyo3::prelude::*;
use pyo3::types::PyDict;
use crate::forcefield;

/// Load a force field from an OpenMM-style XML file
#[pyfunction]
pub fn load_forcefield(path: String) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let ff = forcefield::parse_forcefield_xml(&path).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Force field parsing failed: {}", e))
        })?;

        let dict = PyDict::new_bound(py);

        // Basic info
        dict.set_item("name", &ff.name)?;
        dict.set_item("num_atom_types", ff.atom_types.len())?;
        dict.set_item("num_residue_templates", ff.residue_templates.len())?;
        dict.set_item("num_harmonic_bonds", ff.harmonic_bonds.len())?;
        dict.set_item("num_harmonic_angles", ff.harmonic_angles.len())?;
        dict.set_item("num_proper_torsions", ff.proper_torsions.len())?;
        dict.set_item("num_improper_torsions", ff.improper_torsions.len())?;
        dict.set_item("num_nonbonded_params", ff.nonbonded_params.len())?;
        dict.set_item("num_gbsa_obc_params", ff.gbsa_obc_params.len())?;
        dict.set_item("has_cmap", ff.cmap_data.is_some())?;

        // Atom types as list of dicts
        let mut atom_types = Vec::with_capacity(ff.atom_types.len());
        for at in &ff.atom_types {
            let d = PyDict::new_bound(py);
            d.set_item("name", &at.name).ok();
            d.set_item("class", &at.class).ok();
            d.set_item("element", &at.element).ok();
            d.set_item("mass", at.mass).ok();
            atom_types.push(d.into_py(py));
        }
        dict.set_item("atom_types", atom_types)?;

        // Residue templates
        let mut residue_templates = Vec::with_capacity(ff.residue_templates.len());
        for rt in &ff.residue_templates {
            let d = PyDict::new_bound(py);
            d.set_item("name", &rt.name).ok();
            d.set_item("num_atoms", rt.atoms.len()).ok();
            d.set_item("num_bonds", rt.bonds.len()).ok();

            let mut atoms = Vec::with_capacity(rt.atoms.len());
            for a in &rt.atoms {
                let ad = PyDict::new_bound(py);
                ad.set_item("name", &a.name).ok();
                ad.set_item("type", &a.atom_type).ok();
                ad.set_item("charge", a.charge).ok();
                atoms.push(ad.into_py(py));
            }
            d.set_item("atoms", atoms).ok();

            let mut bonds = Vec::with_capacity(rt.bonds.len());
            for (a1, a2) in &rt.bonds {
                bonds.push((a1.clone(), a2.clone()));
            }
            d.set_item("bonds", bonds).ok();

            residue_templates.push(d.into_py(py));
        }
        dict.set_item("residue_templates", residue_templates)?;

        // Harmonic bonds
        let mut bonds = Vec::with_capacity(ff.harmonic_bonds.len());
        for b in &ff.harmonic_bonds {
            let d = PyDict::new_bound(py);
            d.set_item("class1", &b.class1).ok();
            d.set_item("class2", &b.class2).ok();
            d.set_item("k", b.k).ok();
            d.set_item("length", b.length).ok();
            bonds.push(d.into_py(py));
        }
        dict.set_item("harmonic_bonds", bonds)?;

        // Harmonic angles
        let mut angles = Vec::with_capacity(ff.harmonic_angles.len());
        for a in &ff.harmonic_angles {
            let d = PyDict::new_bound(py);
            d.set_item("class1", &a.class1).ok();
            d.set_item("class2", &a.class2).ok();
            d.set_item("class3", &a.class3).ok();
            d.set_item("k", a.k).ok();
            d.set_item("angle", a.angle).ok();
            angles.push(d.into_py(py));
        }
        dict.set_item("harmonic_angles", angles)?;

        // Proper Torsions
        let mut proper_torsions = Vec::with_capacity(ff.proper_torsions.len());
        for t in &ff.proper_torsions {
            let d = PyDict::new_bound(py);
            d.set_item("class1", &t.class1).ok();
            d.set_item("class2", &t.class2).ok();
            d.set_item("class3", &t.class3).ok();
            d.set_item("class4", &t.class4).ok();

            let mut terms = Vec::with_capacity(t.terms.len());
            for term in &t.terms {
                let td = PyDict::new_bound(py);
                td.set_item("periodicity", term.periodicity).ok();
                td.set_item("phase", term.phase).ok();
                td.set_item("k", term.k).ok();
                terms.push(td.into_py(py));
            }
            d.set_item("terms", terms).ok();
            proper_torsions.push(d.into_py(py));
        }
        dict.set_item("proper_torsions", proper_torsions)?;

        // Improper Torsions
        let mut improper_torsions = Vec::with_capacity(ff.improper_torsions.len());
        for t in &ff.improper_torsions {
            let d = PyDict::new_bound(py);
            d.set_item("class1", &t.class1).ok();
            d.set_item("class2", &t.class2).ok();
            d.set_item("class3", &t.class3).ok();
            d.set_item("class4", &t.class4).ok();

            let mut terms = Vec::with_capacity(t.terms.len());
            for term in &t.terms {
                let td = PyDict::new_bound(py);
                td.set_item("periodicity", term.periodicity).ok();
                td.set_item("phase", term.phase).ok();
                td.set_item("k", term.k).ok();
                terms.push(td.into_py(py));
            }
            d.set_item("terms", terms).ok();
            improper_torsions.push(d.into_py(py));
        }
        dict.set_item("improper_torsions", improper_torsions)?;

        // Nonbonded Params
        let mut nonbonded = Vec::with_capacity(ff.nonbonded_params.len());
        for nb in &ff.nonbonded_params {
            let d = PyDict::new_bound(py);
            d.set_item("atom_type", &nb.atom_type).ok();
            d.set_item("charge", nb.charge).ok();
            d.set_item("sigma", nb.sigma).ok();
            d.set_item("epsilon", nb.epsilon).ok();
            nonbonded.push(d.into_py(py));
        }
        dict.set_item("nonbonded_params", nonbonded)?;

        // GBSA-OBC Params
        let mut gbsa = Vec::with_capacity(ff.gbsa_obc_params.len());
        for g in &ff.gbsa_obc_params {
            let d = PyDict::new_bound(py);
            d.set_item("atom_type", &g.atom_type).ok();
            d.set_item("radius", g.radius).ok();
            d.set_item("scale", g.scale).ok();
            gbsa.push(d.into_py(py));
        }
        dict.set_item("gbsa_obc_params", gbsa)?;

        // CMAP Data
        if let Some(ref cmap) = ff.cmap_data {
            let mut maps = Vec::with_capacity(cmap.maps.len());
            for m in &cmap.maps {
                let d = PyDict::new_bound(py);
                d.set_item("size", m.size).ok();
                d.set_item("energies", m.energies.clone()).ok();
                maps.push(d.into_py(py));
            }
            dict.set_item("cmap_maps", maps)?;

            let mut torsions = Vec::with_capacity(cmap.torsions.len());
            for t in &cmap.torsions {
                let d = PyDict::new_bound(py);
                d.set_item("class1", &t.class1).ok();
                d.set_item("type2", &t.type2).ok();
                d.set_item("type3", &t.type3).ok();
                d.set_item("type4", &t.type4).ok();
                d.set_item("class5", &t.class5).ok();
                d.set_item("map_index", t.map_index).ok();
                torsions.push(d.into_py(py));
            }
            dict.set_item("cmap_torsions", torsions)?;
        }

        Ok(dict.into_py(py))
    })
}
