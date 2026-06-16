use proxide_confind::contact_list::ContactList;
use proxide_confind::coords::ResidueIndex;

#[test]
fn contact_list_default_has_no_pairs() {
    let cl = ContactList::default();
    assert_eq!(cl.pairs.len(), 0, "Default ContactList should have no pairs");
    assert_eq!(cl.degrees.len(), 0, "Default ContactList should have no degrees");
}

#[test]
fn contact_list_degree_lookup() {
    // Create a ContactList with one pair
    let mut cl = ContactList::default();
    cl.pairs.push((ResidueIndex(0), ResidueIndex(1)));
    cl.degrees.push(5.5);
    
    // Test degree lookup both directions (symmetric)
    assert_eq!(cl.degree(ResidueIndex(0), ResidueIndex(1)), Some(5.5));
    assert_eq!(cl.degree(ResidueIndex(1), ResidueIndex(0)), Some(5.5));
}

#[test]
fn contact_list_degree_missing() {
    let mut cl = ContactList::default();
    cl.pairs.push((ResidueIndex(0), ResidueIndex(1)));
    cl.degrees.push(5.5);
    
    // Query a pair not in the list
    assert_eq!(cl.degree(ResidueIndex(0), ResidueIndex(2)), None);
    assert_eq!(cl.degree(ResidueIndex(5), ResidueIndex(6)), None);
}

#[test]
fn contact_list_pairs_and_degrees_match_length() {
    let mut cl = ContactList::default();
    
    // Add several pairs
    cl.pairs.push((ResidueIndex(0), ResidueIndex(1)));
    cl.pairs.push((ResidueIndex(1), ResidueIndex(2)));
    cl.pairs.push((ResidueIndex(2), ResidueIndex(3)));
    
    cl.degrees.push(1.0);
    cl.degrees.push(2.0);
    cl.degrees.push(3.0);
    
    assert_eq!(cl.pairs.len(), cl.degrees.len());
    assert_eq!(cl.pairs.len(), 3);
}

#[test]
fn contact_list_ordered_pairs_if_exists() {
    let mut cl = ContactList::default();
    
    // Add pairs in non-ascending order
    cl.pairs.push((ResidueIndex(2), ResidueIndex(3)));
    cl.pairs.push((ResidueIndex(0), ResidueIndex(1)));
    cl.pairs.push((ResidueIndex(1), ResidueIndex(2)));
    
    cl.degrees.push(1.0);
    cl.degrees.push(2.0);
    cl.degrees.push(3.0);
    
    let ordered = cl.ordered_pairs();
    
    // Check that ordered pairs are in ascending order
    assert_eq!(ordered[0], (ResidueIndex(0), ResidueIndex(1)));
    assert_eq!(ordered[1], (ResidueIndex(1), ResidueIndex(2)));
    assert_eq!(ordered[2], (ResidueIndex(2), ResidueIndex(3)));
}
