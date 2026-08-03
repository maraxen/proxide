use proxide_rotlib::{counts_as_sidechain, is_backbone_or_hydrogen};

const NON_ALA: &[&str] = &[
    "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "SER",
    "THR", "TRP", "TYR", "VAL",
];

#[test]
fn test_counts_cb_ala_true() {
    assert!(counts_as_sidechain("CB", "ALA"));
}

#[test]
fn test_counts_cb_arg_false() {
    assert!(!counts_as_sidechain("CB", "ARG"));
}

#[test]
fn test_counts_cb_all_non_ala_false() {
    for &aa in NON_ALA {
        assert!(!counts_as_sidechain("CB", aa), "CB/{aa} should be false");
    }
}

#[test]
fn test_counts_backbone_n_false() {
    assert!(!counts_as_sidechain("N", "ALA"));
}

#[test]
fn test_counts_backbone_ca_false() {
    assert!(!counts_as_sidechain("CA", "ALA"));
}

#[test]
fn test_counts_backbone_c_false() {
    assert!(!counts_as_sidechain("C", "ALA"));
}

#[test]
fn test_counts_backbone_o_false() {
    assert!(!counts_as_sidechain("O", "ARG"));
}

#[test]
fn test_counts_hydrogen_prefix_false() {
    assert!(!counts_as_sidechain("HB2", "LEU"));
}

#[test]
fn test_counts_hydrogen_bare_h_false() {
    assert!(!counts_as_sidechain("H", "ALA"));
}

#[test]
fn test_counts_sidechain_heavy_true() {
    assert!(counts_as_sidechain("CD", "LEU"));
}

#[test]
fn test_counts_sidechain_heavy_arg() {
    assert!(counts_as_sidechain("NE", "ARG"));
}

#[test]
fn test_is_backbone_n() {
    assert!(is_backbone_or_hydrogen("N"));
}

#[test]
fn test_is_backbone_ca() {
    assert!(is_backbone_or_hydrogen("CA"));
}

#[test]
fn test_is_backbone_c() {
    assert!(is_backbone_or_hydrogen("C"));
}

#[test]
fn test_is_backbone_o() {
    assert!(is_backbone_or_hydrogen("O"));
}

#[test]
fn test_is_backbone_h() {
    assert!(is_backbone_or_hydrogen("H"));
}

#[test]
fn test_is_backbone_h_prefix() {
    assert!(is_backbone_or_hydrogen("HB2"));
}

#[test]
fn test_is_not_backbone_cd() {
    assert!(!is_backbone_or_hydrogen("CD"));
}

#[test]
fn test_is_not_backbone_cb() {
    assert!(!is_backbone_or_hydrogen("CB"));
}

#[test]
fn test_is_not_backbone_empty() {
    assert!(!is_backbone_or_hydrogen(""));
}
