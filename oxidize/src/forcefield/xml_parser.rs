//! XML parser for OpenMM-style force field files.
//!
//! Parses force field XML files (e.g., protein.ff19SB.xml) into
//! the ForceField data structure.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;
use thiserror::Error;

use super::types::*;

/// Errors that can occur during force field parsing
#[derive(Error, Debug)]
pub enum ParseError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("XML parsing error: {0}")]
    Xml(#[from] quick_xml::Error),

    #[error("Missing required attribute: {0}")]
    MissingAttribute(String),

    #[error("Invalid attribute value: {0}")]
    InvalidValue(String),

    #[error("UTF-8 decode error: {0}")]
    Utf8(#[from] std::str::Utf8Error),

    #[error("Feature not implemented: {0}")]
    NotImplemented(String),
}

/// Parse a force field XML file
pub fn parse_forcefield_xml(path: &str) -> Result<ForceField, ParseError> {
    let path = Path::new(path);
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    parse_xml_reader(reader, name)
}

/// Parse from any reader
fn parse_xml_reader<R: std::io::BufRead>(
    reader: R,
    name: String,
) -> Result<ForceField, ParseError> {
    let mut xml_reader = Reader::from_reader(reader);
    xml_reader.trim_text(true);

    let mut ff = ForceField::new(name);
    let mut buf = Vec::new();

    loop {
        match xml_reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => match e.name().as_ref() {
                b"AtomTypes" => parse_atom_types(&mut xml_reader, &mut ff)?,
                b"Residues" => parse_residues(&mut xml_reader, &mut ff)?,
                b"HarmonicBondForce" => parse_harmonic_bonds(&mut xml_reader, &mut ff)?,
                b"HarmonicAngleForce" => parse_harmonic_angles(&mut xml_reader, &mut ff)?,
                b"PeriodicTorsionForce" => parse_periodic_torsions(&mut xml_reader, &mut ff)?,
                b"NonbondedForce" => parse_nonbonded(&mut xml_reader, &mut ff)?,
                b"GBSAOBCForce" => parse_gbsa_obc(&mut xml_reader, &mut ff)?,
                b"CMAPTorsionForce" => parse_cmap(&mut xml_reader, &mut ff)?,
                b"AmoebaMultipoleForce" | b"MultipoleForce" => {
                    return Err(ParseError::NotImplemented(
                        "MultipoleForce (AMOEBA) is not yet supported".to_string(),
                    ));
                }
                b"AmoebaVdwForce" => {
                    return Err(ParseError::NotImplemented(
                        "AmoebaVdwForce is not yet supported".to_string(),
                    ));
                }
                _ => {}
            },
            Ok(Event::Eof) => break,
            Err(e) => return Err(ParseError::Xml(e)),
            _ => {}
        }
        buf.clear();
    }

    ff.build_indices();
    Ok(ff)
}

/// Helper to get a required attribute as string
fn get_attr(e: &BytesStart, name: &[u8]) -> Result<String, ParseError> {
    for attr in e.attributes().flatten() {
        if attr.key.as_ref() == name {
            return Ok(String::from_utf8_lossy(&attr.value).to_string());
        }
    }
    Err(ParseError::MissingAttribute(
        String::from_utf8_lossy(name).to_string(),
    ))
}

/// Helper to get an optional attribute as string
fn get_attr_opt(e: &BytesStart, name: &[u8]) -> Option<String> {
    for attr in e.attributes().flatten() {
        if attr.key.as_ref() == name {
            return Some(String::from_utf8_lossy(&attr.value).to_string());
        }
    }
    None
}

/// Helper to get one of two attributes (e.g. "class1" or "type1")
fn get_attr_either(e: &BytesStart, name1: &[u8], name2: &[u8]) -> Result<String, ParseError> {
    get_attr_opt(e, name1)
        .or_else(|| get_attr_opt(e, name2))
        .ok_or_else(|| {
            ParseError::MissingAttribute(format!(
                "{} or {}",
                String::from_utf8_lossy(name1),
                String::from_utf8_lossy(name2)
            ))
        })
}

/// Helper to parse f32 attribute
fn get_attr_f32(e: &BytesStart, name: &[u8]) -> Result<f32, ParseError> {
    let val = get_attr(e, name)?;
    val.parse::<f32>()
        .map_err(|_| ParseError::InvalidValue(format!("Cannot parse '{}' as f32", val)))
}

/// Helper to parse optional f32 attribute
fn get_attr_f32_opt(e: &BytesStart, name: &[u8]) -> Option<f32> {
    get_attr_opt(e, name).and_then(|v| v.parse().ok())
}

/// Helper to parse u32 attribute
fn get_attr_u32(e: &BytesStart, name: &[u8]) -> Result<u32, ParseError> {
    let val = get_attr(e, name)?;
    val.parse::<u32>()
        .map_err(|_| ParseError::InvalidValue(format!("Cannot parse '{}' as u32", val)))
}

/// Parse <AtomTypes> section
fn parse_atom_types<R: std::io::BufRead>(
    reader: &mut Reader<R>,
    ff: &mut ForceField,
) -> Result<(), ParseError> {
    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Empty(ref e)) if e.name().as_ref() == b"Type" => {
                let atom_type = AtomType {
                    name: get_attr(e, b"name")?,
                    class: get_attr_either(e, b"class", b"type")
                        .unwrap_or_else(|_| get_attr(e, b"name").unwrap_or_default()),
                    element: get_attr_opt(e, b"element").unwrap_or_default(),
                    mass: get_attr_f32(e, b"mass")?,
                    charge: get_attr_f32_opt(e, b"charge"),
                };
                ff.atom_types.push(atom_type);
            }
            Ok(Event::End(ref e)) if e.name().as_ref() == b"AtomTypes" => break,
            Ok(Event::Eof) => break,
            Err(e) => return Err(ParseError::Xml(e)),
            _ => {}
        }
        buf.clear();
    }

    Ok(())
}

/// Parse <Residues> section
fn parse_residues<R: std::io::BufRead>(
    reader: &mut Reader<R>,
    ff: &mut ForceField,
) -> Result<(), ParseError> {
    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) if e.name().as_ref() == b"Residue" => {
                let template = parse_single_residue(reader, e)?;
                ff.residue_templates.push(template);
            }
            Ok(Event::End(ref e)) if e.name().as_ref() == b"Residues" => break,
            Ok(Event::Eof) => break,
            Err(e) => return Err(ParseError::Xml(e)),
            _ => {}
        }
        buf.clear();
    }

    Ok(())
}

/// Parse a single <Residue> element
fn parse_single_residue<R: std::io::BufRead>(
    reader: &mut Reader<R>,
    start: &BytesStart,
) -> Result<ResidueTemplate, ParseError> {
    let name = get_attr(start, b"name")?;
    let override_level = get_attr_opt(start, b"override").and_then(|v| v.parse().ok());

    let mut atoms = Vec::new();
    let mut bonds = Vec::new();
    let mut external_bonds = Vec::new();

    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Empty(ref e)) => match e.name().as_ref() {
                b"Atom" => {
                    atoms.push(ResidueAtom {
                        name: get_attr(e, b"name")?,
                        atom_type: get_attr(e, b"type")?,
                        charge: get_attr_f32_opt(e, b"charge"),
                    });
                }
                b"Bond" => {
                    // Try "atomName1"/"atomName2" first
                    let n1 = get_attr_opt(e, b"atomName1");
                    let n2 = get_attr_opt(e, b"atomName2");

                    if let (Some(n1), Some(n2)) = (n1, n2) {
                        bonds.push((n1, n2));
                    } else {
                        // Try "from"/"to" (indices or names)
                        let from = get_attr(e, b"from")?;
                        let to = get_attr(e, b"to")?;

                        // Try to parse as indices
                        let name1 = if let Ok(idx) = from.parse::<usize>() {
                            atoms.get(idx).map(|a| a.name.clone()).ok_or_else(|| {
                                ParseError::InvalidValue(format!(
                                    "Bond index {} out of bounds",
                                    idx
                                ))
                            })?
                        } else {
                            from
                        };

                        let name2 = if let Ok(idx) = to.parse::<usize>() {
                            atoms.get(idx).map(|a| a.name.clone()).ok_or_else(|| {
                                ParseError::InvalidValue(format!(
                                    "Bond index {} out of bounds",
                                    idx
                                ))
                            })?
                        } else {
                            to
                        };

                        bonds.push((name1, name2));
                    }
                }
                b"ExternalBond" => {
                    // Try "atomName" first
                    if let Some(name) = get_attr_opt(e, b"atomName") {
                        external_bonds.push(name);
                    } else {
                        // Try "from" (index or name)
                        let from = get_attr(e, b"from")?;

                        // Try to parse as index
                        let name = if let Ok(idx) = from.parse::<usize>() {
                            atoms.get(idx).map(|a| a.name.clone()).ok_or_else(|| {
                                ParseError::InvalidValue(format!(
                                    "ExternalBond index {} out of bounds",
                                    idx
                                ))
                            })?
                        } else {
                            from
                        };

                        external_bonds.push(name);
                    }
                }
                _ => {}
            },
            Ok(Event::End(ref e)) if e.name().as_ref() == b"Residue" => break,
            Ok(Event::Eof) => break,
            Err(e) => return Err(ParseError::Xml(e)),
            _ => {}
        }
        buf.clear();
    }

    Ok(ResidueTemplate {
        name,
        atoms,
        bonds,
        external_bonds,
        override_level,
    })
}

/// Parse <HarmonicBondForce> section
fn parse_harmonic_bonds<R: std::io::BufRead>(
    reader: &mut Reader<R>,
    ff: &mut ForceField,
) -> Result<(), ParseError> {
    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Empty(ref e)) if e.name().as_ref() == b"Bond" => {
                ff.harmonic_bonds.push(HarmonicBondParam {
                    class1: get_attr_either(e, b"class1", b"type1")?,
                    class2: get_attr_either(e, b"class2", b"type2")?,
                    k: get_attr_f32(e, b"k")?,
                    length: get_attr_f32(e, b"length")?,
                });
            }
            Ok(Event::End(ref e)) if e.name().as_ref() == b"HarmonicBondForce" => break,
            Ok(Event::Eof) => break,
            Err(e) => return Err(ParseError::Xml(e)),
            _ => {}
        }
        buf.clear();
    }

    Ok(())
}

/// Parse <HarmonicAngleForce> section
fn parse_harmonic_angles<R: std::io::BufRead>(
    reader: &mut Reader<R>,
    ff: &mut ForceField,
) -> Result<(), ParseError> {
    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Empty(ref e)) if e.name().as_ref() == b"Angle" => {
                ff.harmonic_angles.push(HarmonicAngleParam {
                    class1: get_attr_either(e, b"class1", b"type1")?,
                    class2: get_attr_either(e, b"class2", b"type2")?,
                    class3: get_attr_either(e, b"class3", b"type3")?,
                    k: get_attr_f32(e, b"k")?,
                    angle: get_attr_f32(e, b"angle")?,
                });
            }
            Ok(Event::End(ref e)) if e.name().as_ref() == b"HarmonicAngleForce" => break,
            Ok(Event::Eof) => break,
            Err(e) => return Err(ParseError::Xml(e)),
            _ => {}
        }
        buf.clear();
    }

    Ok(())
}

/// Parse <PeriodicTorsionForce> section
fn parse_periodic_torsions<R: std::io::BufRead>(
    reader: &mut Reader<R>,
    ff: &mut ForceField,
) -> Result<(), ParseError> {
    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Empty(ref e)) => match e.name().as_ref() {
                b"Proper" => {
                    let torsion = parse_torsion_element(e)?;
                    ff.proper_torsions.push(ProperTorsionParam {
                        class1: torsion.0,
                        class2: torsion.1,
                        class3: torsion.2,
                        class4: torsion.3,
                        terms: torsion.4,
                    });
                }
                b"Improper" => {
                    let torsion = parse_torsion_element(e)?;
                    ff.improper_torsions.push(ImproperTorsionParam {
                        class1: torsion.0,
                        class2: torsion.1,
                        class3: torsion.2,
                        class4: torsion.3,
                        terms: torsion.4,
                    });
                }
                _ => {}
            },
            Ok(Event::End(ref e)) if e.name().as_ref() == b"PeriodicTorsionForce" => break,
            Ok(Event::Eof) => break,
            Err(e) => return Err(ParseError::Xml(e)),
            _ => {}
        }
        buf.clear();
    }

    Ok(())
}

/// Parse a torsion element with potentially multiple periodicities
fn parse_torsion_element(
    e: &BytesStart,
) -> Result<(String, String, String, String, Vec<TorsionTerm>), ParseError> {
    let class1 = get_attr_opt(e, b"class1")
        .or(get_attr_opt(e, b"type1"))
        .unwrap_or_default();
    let class2 = get_attr_opt(e, b"class2")
        .or(get_attr_opt(e, b"type2"))
        .unwrap_or_default();
    let class3 = get_attr_opt(e, b"class3")
        .or(get_attr_opt(e, b"type3"))
        .unwrap_or_default();
    let class4 = get_attr_opt(e, b"class4")
        .or(get_attr_opt(e, b"type4"))
        .unwrap_or_default();

    let mut terms = Vec::new();

    // Check for single term (periodicity, phase, k)
    if let (Some(periodicity), Some(phase), Some(k)) = (
        get_attr_f32_opt(e, b"periodicity"),
        get_attr_f32_opt(e, b"phase"),
        get_attr_f32_opt(e, b"k"),
    ) {
        terms.push(TorsionTerm {
            periodicity: periodicity as u32,
            phase,
            k,
        });
    }

    // Check for multiple terms (periodicity1, phase1, k1, etc.)
    for i in 1..=6 {
        let per_key = format!("periodicity{}", i);
        let phase_key = format!("phase{}", i);
        let k_key = format!("k{}", i);

        if let (Some(periodicity), Some(phase), Some(k)) = (
            get_attr_f32_opt(e, per_key.as_bytes()),
            get_attr_f32_opt(e, phase_key.as_bytes()),
            get_attr_f32_opt(e, k_key.as_bytes()),
        ) {
            terms.push(TorsionTerm {
                periodicity: periodicity as u32,
                phase,
                k,
            });
        }
    }

    Ok((class1, class2, class3, class4, terms))
}

/// Parse <NonbondedForce> section
///
/// Note: In some force fields (like ff19SB), atoms use "class" instead of "type",
/// and charges are defined per-residue rather than per-atom-type.
fn parse_nonbonded<R: std::io::BufRead>(
    reader: &mut Reader<R>,
    ff: &mut ForceField,
) -> Result<(), ParseError> {
    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Empty(ref e)) if e.name().as_ref() == b"Atom" => {
                // Try "type" first, fall back to "class"
                let atom_type = get_attr_opt(e, b"type")
                    .or_else(|| get_attr_opt(e, b"class"))
                    .ok_or_else(|| ParseError::MissingAttribute("type or class".to_string()))?;

                // Charge is optional (may be defined per-residue)
                let charge = get_attr_f32_opt(e, b"charge").unwrap_or(0.0);

                ff.nonbonded_params.push(NonbondedParam {
                    atom_type,
                    charge,
                    sigma: get_attr_f32(e, b"sigma")?,
                    epsilon: get_attr_f32(e, b"epsilon")?,
                });
            }
            Ok(Event::End(ref e)) if e.name().as_ref() == b"NonbondedForce" => break,
            Ok(Event::Eof) => break,
            Err(e) => return Err(ParseError::Xml(e)),
            _ => {}
        }
        buf.clear();
    }

    Ok(())
}

/// Parse <GBSAOBCForce> section for implicit solvent
fn parse_gbsa_obc<R: std::io::BufRead>(
    reader: &mut Reader<R>,
    ff: &mut ForceField,
) -> Result<(), ParseError> {
    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Empty(ref e)) if e.name().as_ref() == b"Atom" => {
                ff.gbsa_obc_params.push(GBSAOBCParam {
                    atom_type: get_attr(e, b"type")?,
                    radius: get_attr_f32(e, b"radius")?,
                    scale: get_attr_f32(e, b"scale")?,
                });
            }
            Ok(Event::End(ref e)) if e.name().as_ref() == b"GBSAOBCForce" => break,
            Ok(Event::Eof) => break,
            Err(e) => return Err(ParseError::Xml(e)),
            _ => {}
        }
        buf.clear();
    }

    Ok(())
}

/// Parse <CMAPTorsionForce> section
fn parse_cmap<R: std::io::BufRead>(
    reader: &mut Reader<R>,
    ff: &mut ForceField,
) -> Result<(), ParseError> {
    let mut buf = Vec::new();
    let mut maps = Vec::new();
    let mut torsions = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) if e.name().as_ref() == b"Map" => {
                // Read the grid data as text content
                let grid = parse_cmap_grid(reader)?;
                maps.push(grid);
            }
            Ok(Event::Empty(ref e)) if e.name().as_ref() == b"Torsion" => {
                torsions.push(CMAPTorsion {
                    class1: get_attr_either(e, b"class1", b"type1")?,
                    type2: get_attr_either(e, b"class2", b"type2")?,
                    type3: get_attr_either(e, b"class3", b"type3")?,
                    type4: get_attr_either(e, b"class4", b"type4")?,
                    class5: get_attr_either(e, b"class5", b"type5")?,
                    map_index: get_attr_u32(e, b"map")? as usize,
                });
            }
            Ok(Event::End(ref e)) if e.name().as_ref() == b"CMAPTorsionForce" => break,
            Ok(Event::Eof) => break,
            Err(e) => return Err(ParseError::Xml(e)),
            _ => {}
        }
        buf.clear();
    }

    if !maps.is_empty() {
        ff.cmap_data = Some(CMAPData { maps, torsions });
    }

    Ok(())
}

/// Parse CMAP grid content
fn parse_cmap_grid<R: std::io::BufRead>(reader: &mut Reader<R>) -> Result<CMAPGrid, ParseError> {
    let mut buf = Vec::new();
    let mut text_content = String::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Text(e)) => {
                text_content.push_str(&e.unescape().unwrap_or_default());
            }
            Ok(Event::End(ref e)) if e.name().as_ref() == b"Map" => break,
            Ok(Event::Eof) => break,
            Err(e) => return Err(ParseError::Xml(e)),
            _ => {}
        }
        buf.clear();
    }

    // Parse space-separated floats
    let energies: Vec<f32> = text_content
        .split_whitespace()
        .filter_map(|s| s.parse().ok())
        .collect();

    // Determine grid size (typically 24x24 = 576)
    let size = (energies.len() as f64).sqrt() as usize;

    Ok(CMAPGrid { size, energies })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_atom_types_xml() {
        let xml = r#"<?xml version="1.0"?>
        <ForceField>
          <AtomTypes>
            <Type class="protein-C" element="C" mass="12.01" name="protein-C" />
            <Type class="protein-N" element="N" mass="14.01" name="protein-N" />
          </AtomTypes>
        </ForceField>"#;

        let ff = parse_xml_reader(xml.as_bytes(), "test".to_string()).unwrap();

        assert_eq!(ff.atom_types.len(), 2);
        assert_eq!(ff.atom_types[0].name, "protein-C");
        assert_eq!(ff.atom_types[0].mass, 12.01);
        assert_eq!(ff.atom_types[1].element, "N");
    }

    #[test]
    fn test_parse_residue_xml() {
        let xml = r#"<?xml version="1.0"?>
        <ForceField>
          <Residues>
            <Residue name="ALA">
              <Atom charge="-0.4157" name="N" type="protein-N" />
              <Atom charge="0.2719" name="H" type="protein-H" />
              <Bond atomName1="N" atomName2="H" />
              <ExternalBond atomName="N" />
            </Residue>
          </Residues>
        </ForceField>"#;

        let ff = parse_xml_reader(xml.as_bytes(), "test".to_string()).unwrap();

        assert_eq!(ff.residue_templates.len(), 1);
        assert_eq!(ff.residue_templates[0].name, "ALA");
        assert_eq!(ff.residue_templates[0].atoms.len(), 2);
        assert_eq!(ff.residue_templates[0].bonds.len(), 1);
        assert_eq!(ff.residue_templates[0].external_bonds.len(), 1);
    }

    #[test]
    fn test_parse_harmonic_bond_xml() {
        let xml = r#"<?xml version="1.0"?>
        <ForceField>
          <HarmonicBondForce>
            <Bond class1="protein-C" class2="protein-N" k="410031.0" length="0.1335" />
          </HarmonicBondForce>
        </ForceField>"#;

        let ff = parse_xml_reader(xml.as_bytes(), "test".to_string()).unwrap();

        assert_eq!(ff.harmonic_bonds.len(), 1);
        assert_eq!(ff.harmonic_bonds[0].k, 410031.0);
        assert_eq!(ff.harmonic_bonds[0].length, 0.1335);
    }

    #[test]
    fn test_parse_angle_xml() {
        let xml = r#"<?xml version="1.0"?>
        <ForceField>
          <HarmonicAngleForce>
            <Angle angle="2.0943951" class1="C" class2="CA" class3="CA" k="527.184" />
          </HarmonicAngleForce>
        </ForceField>"#;

        let ff = parse_xml_reader(xml.as_bytes(), "test".to_string()).unwrap();

        assert_eq!(ff.harmonic_angles.len(), 1);
        assert!((ff.harmonic_angles[0].angle - 2.0943951).abs() < 1e-5);
    }
}
