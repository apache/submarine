package org.zapodot.junit.ldap;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

public class EmbeddedLdapRuleBuilderTest {

    @Test
    public void bindingToLegalPort() {
        assertNotNull(EmbeddedLdapRuleBuilder.newInstance().bindingToPort(9999));
    }

    @Test(expected = IllegalStateException.class)
    public void testPrematureLdapConnection() throws Exception {
        EmbeddedLdapRuleBuilder.newInstance().build().ldapConnection();

    }

    @Test(expected = IllegalStateException.class)
    public void testPrematureContext() throws Exception {
        EmbeddedLdapRuleBuilder.newInstance().build().context();

    }

    @Test(expected = IllegalArgumentException.class)
    public void testUnknownLDIF() {
        EmbeddedLdapRuleBuilder.newInstance().importingLdifs("nonExisting.ldif").build();

    }

    @Test
    public void testNullLDIF() {
        assertNotNull(EmbeddedLdapRuleBuilder.newInstance().importingLdifs(null).build());

    }

    @Test(expected = IllegalStateException.class)
    public void testIllegalDSN() {
        EmbeddedLdapRuleBuilder.newInstance().usingBindDSN("bindDsn").build();

    }

    @Test(expected = IllegalArgumentException.class)
    public void testIllegalPort() {
        EmbeddedLdapRuleBuilder.newInstance().bindingToPort(Integer.MIN_VALUE).build();

    }

    @Test(expected = IllegalArgumentException.class)
    public void testSchemaNotFound() {
        EmbeddedLdapRuleBuilder.newInstance().withSchema("non-existing-schema.ldif").build();

    }

    @Test(expected = IllegalArgumentException.class)
    public void testSchemaIsNotAFile() {
        EmbeddedLdapRuleBuilder.newInstance().withSchema("folder").build();

    }

    @Test(expected = IllegalArgumentException.class)
    public void testSchemaIsInvalid() {
        EmbeddedLdapRuleBuilder.newInstance().withSchema("invalid.ldif").build();

    }

    @Test(expected = IllegalArgumentException.class)
    public void testSchemaFileUnsupportedIsInvalid() {
        EmbeddedLdapRuleBuilder.newInstance().withSchema("\"#%¤&&%/¤##¤¤").build();

    }

    @Test(expected = IllegalArgumentException.class)
    public void testInvalidPort() {
        EmbeddedLdapRuleBuilder.newInstance().bindingToPort(Integer.MAX_VALUE);

    }

    @Test(expected = IllegalArgumentException.class)
    public void testInvalidBindAddress() {
        EmbeddedLdapRuleBuilder.newInstance().bindingToAddress("åpsldfåpl");

    }


}