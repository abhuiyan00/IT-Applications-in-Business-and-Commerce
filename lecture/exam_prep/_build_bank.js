// One-shot generator: rebuilds chapters.js from two curated sources.
//  (1) Study sets  -> chapters.backup.js (pristine), de-biased (option order shuffled).
//  (2) Testownik    -> the curated EXAM array below: real past-exam concepts, DEDUPLICATED
//      (the raw bank uploads/Quiz_Question_Bank.md had ~169 items with the same concept
//      repeated up to 6×; collapsed to one well-formed question each), every answer
//      verified against uploads/ITABC.md, normalized to single-correct / 4 options.
// Run: node _build_bank.js   (writes chapters.js)
//
// To change study questions edit chapters.backup.js; to change exam questions edit the
// EXAM array here. Then re-run this script. It assigns global ids and de-biases option
// order deterministically (fixed seed) so the answer is never guessable by position.
const fs = require('fs');

// deterministic RNG so re-runs are stable
let seed = 1337;
function rnd() { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; }
function shuffle(a) { a = a.slice(); for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; }

// ---- load synthetic bank (ALWAYS from the pristine backup, never the generated file) ----
eval(fs.readFileSync('chapters.backup.js', 'utf8').replace('const quizData =', 'globalThis.quizData ='));
const studySets = quizData.sets.map(s => ({
  id: s.id, title: s.title, group: 'study',
  questions: s.questions.map(q => {
    // shuffle option order, recompute correct index -> kills positional tell
    const order = shuffle(q.options.map((_, i) => i));
    const options = order.map(i => q.options[i]);
    const correct = order.indexOf(q.correct);
    return { question: q.question, options, correct, reasoning: q.reasoning };
  })
}));

// ---------------------------------------------------------------------------
// Curated, de-duplicated Testownik bank.
// Each entry: { q, o:[4 strings], c:<index of the correct one>, r:reasoning }.
// `c` indexes the option AS WRITTEN; the builder shuffles options per question.
// Where the raw bank's answer key was wrong it has been corrected (see notes):
//   * "manage only apps+data" = PaaS  (raw said SaaS; ITABC §10.16 table: in PaaS the
//     user manages Application+Data only, in SaaS the provider manages those too).
//   * HTTPS cert hostname check = CN/SAN vs DNS  (not DN vs IP) per X.509/RFC 6125.
//   * Proof-of-Work header uses the Merkle root (not a raw transaction hash).
//   * Hash collision = SAME hash for two different messages.
// ---------------------------------------------------------------------------
const EXAM = {
  // ===== 21 · Web Services & SOA =====
  WS: { title: '21 · Testownik — Web Services & SOA', q: [
    { q: 'BPEL (Business Process Execution Language) is:',
      o: ['An XML-based, executable language for orchestrating business processes from web services', 'A graphical notation for drawing business processes', 'A language for describing service choreography', 'A description of a network service’s communication interface'], c: 0,
      r: 'BPEL is an XML-based, executable orchestration language for composing web services. BPMN draws processes; WSDL describes the interface.' },
    { q: 'BPMN is:',
      o: ['A graphical notation for modelling business processes (Business Process Modeling Notation)', 'An XML language for defining business processes in web services', 'A choreography language based on PI-Calculus', 'A binary RPC wire format'], c: 0,
      r: 'BPMN = Business Process Modeling Notation — a standardized graphical notation for drawing business processes.' },
    { q: 'Which statement about a SOAP message is TRUE?',
      o: ['It must be XML with a mandatory Envelope and Body, use namespaces, and may have an optional Header', 'It must reference a DTD', 'It is transmitted as binary rather than XML', 'Its Header element is mandatory'], c: 0,
      r: 'A SOAP message is XML with a mandatory Envelope+Body (Header optional), uses namespaces, and must not reference DTDs. SOAP is a W3C standard.' },
    { q: 'What is WSDL used for (the acronym for the XML language describing how to talk to a network service)?',
      o: ['Describing what a service offers, where it is located, and how to call it', 'Encrypting the body of SOAP messages', 'Discovering services in a public registry', 'Balancing load across service nodes'], c: 0,
      r: 'WSDL (Web Services Description Language) is the machine-readable contract: operations, location and message formats. (UDDI is the registry; WSDL the interface.)' },
    { q: 'Which statement about XML is true?',
      o: ['Every tag must have a closing tag', 'All tags must be lowercase', 'Every document must have a DTD', 'Tags may overlap (improper nesting is allowed)'], c: 0,
      r: 'XML is strict: every element must be closed and properly nested. Case matters but need not be lowercase; a DTD is optional.' },
    { q: 'XML Schema (XSD) is used for:',
      o: ['Defining the allowed structure and data types of an XML document', 'Transforming one XML document into another', 'Encrypting an XML document', 'Rendering XML directly as HTML'], c: 0,
      r: 'XSD defines the structure/types an XML document must follow (XSLT transforms; it is a separate language).' },
    { q: 'Simple-type elements in XML Schema are elements that:',
      o: ['Cannot contain attributes or other elements (text content only)', 'Can contain attributes but not other elements', 'Can contain other elements but not attributes', 'Must contain at least one child element'], c: 0,
      r: 'A simple-type element carries only text — no attributes and no child elements; anything with attributes/children is a complex type.' },
    { q: 'Which set of properties correctly describes XML?',
      o: ['Text-based, an accepted standard, with simple rules', 'Binary and proprietary', 'A draft that is not yet an accepted standard', 'Deliberately complicated rules'], c: 0,
      r: 'XML is a text-based, widely accepted W3C standard built on a small set of simple rules.' },
    { q: 'CORBA technology:',
      o: ['Enables communication between objects running in heterogeneous systems', 'Is a graphical notation for business processes', 'Describes service choreography based on PI-Calculus', 'Is a relational database query language'], c: 0,
      r: 'CORBA is middleware (an ORB) that lets objects in different languages/platforms invoke each other.' },
    { q: 'CORBA stands for:',
      o: ['Common Object Request Broker Architecture', 'Common Object Related Binding Architecture', 'Common Object Relative Business Architecture', 'Centralized Object Routing Backbone Application'], c: 0,
      r: 'CORBA = Common Object Request Broker Architecture (OMG standard).' },
    { q: 'Which are known problems of CORBA?',
      o: ['Interfaces must be written in IDL, there is no generalized security model, and firewall traversal is difficult', 'It has no language independence', 'It requires JSON payloads', 'It runs only on Windows'], c: 0,
      r: 'CORBA needs IDL interface descriptions, lacks a generalized security model, and struggles to pass through firewalls.' },
    { q: 'CORBA serializes data on the wire using:',
      o: ['IIOP', 'JSON', 'XML', 'Protocol Buffers'], c: 0,
      r: 'CORBA uses IIOP (Internet Inter-ORB Protocol) for serialization/transport.' },
    { q: 'What is IDL?',
      o: ['A language for describing the communication interface between a client and a server (Interface Definition Language)', 'A Java class that builds interfaces at runtime', 'A C++ template library', 'A relational schema definition'], c: 0,
      r: 'IDL (Interface Definition Language) describes the client–server interface, language-independently; it is compiled into stubs/skeletons.' },
    { q: 'What is marshalling?',
      o: ['Converting data from a language-dependent format to a language-independent one for transport', 'Defining binary values for a decimal string', 'Quantizing a data sample into a client format', 'Building an interface framework'], c: 0,
      r: 'Marshalling serializes in-memory (language-dependent) data into a portable wire format; unmarshalling reverses it.' },
    { q: 'A Java RMI remote interface must:',
      o: ['Be public, extend java.rmi.Remote, and declare RemoteException on every method', 'Be a final class implementing no interface', 'Avoid extending java.rmi.Remote', 'Declare no exceptions at all'], c: 0,
      r: 'RMI remote interfaces are public, extend java.rmi.Remote, and every method declares java.rmi.RemoteException.' },
    { q: 'Which servers are compatible with Java EE?',
      o: ['JOnAS and Oracle WebLogic Server', 'ZendServer and WebSphere', 'JBoss and Microsoft Application Server', 'Apache HTTP Server and nginx'], c: 0,
      r: 'Java EE application servers include JOnAS, Oracle WebLogic, JBoss, WebSphere, GlassFish (Apache HTTP/nginx are web servers, not Java EE app servers).' },
    { q: 'Which statement about JBoss is TRUE?',
      o: ['It is a 100% Java, LGPL-licensed Java EE application server based on EJB, integrated with Eclipse', 'It is written mostly in C++', 'It is proprietary, closed-source software', 'It cannot be integrated with Eclipse'], c: 0,
      r: 'JBoss is a 100% Java, LGPL, EJB-based application server with Eclipse tooling.' },
    { q: 'Which statement about Java Servlets is TRUE?',
      o: ['They are Java classes that run inside a web container to handle HTTP requests', 'They run outside any container as standalone processes', 'They are written in C++ for performance', 'They remove the need for an application server'], c: 0,
      r: 'A servlet is a Java class executing inside a web container, with methods to handle HTTP requests.' },
    { q: 'Which is a real protocol for HTTP-server ↔ application-server communication?',
      o: ['AJP13 (Apache JServ Protocol, e.g. Apache↔Tomcat)', 'DEV3 “Devoted Transit 3”', 'SMTP relay handover', 'FTP passive mode'], c: 0,
      r: 'AJP13 (and plain HTTP, e.g. to WebSphere) connect the web server to the application server; “DEV3” is invented.' },
    { q: 'NuSOAP is a SOAP library for:',
      o: ['PHP', 'Java', 'C++', 'C#'], c: 0,
      r: 'NuSOAP is a PHP library for building/consuming SOAP web services.' },
    { q: 'UDDI is:',
      o: ['A registry of XML-based descriptions of web services', 'A choreography language based on PI-Calculus', 'A graphical notation for business processes', 'A transport-layer encryption protocol'], c: 0,
      r: 'UDDI is the directory/registry where web-service descriptions are published and discovered.' },
    { q: 'Microservices are:',
      o: ['A fine-grained variant of SOA — small, independently deployable services', 'A containerization variant used only for web services', 'An asynchronous messaging mechanism', 'A type of bare-metal hypervisor'], c: 0,
      r: 'Microservices are an SOA variant: loosely-coupled, fine-grained, independently deployable services over lightweight protocols.' },
    { q: 'A Python library for building REST servers is:',
      o: ['AIOHTTP', 'JAX-RS', 'AXIS', 'Hibernate'], c: 0,
      r: 'AIOHTTP (also Flask/FastAPI) builds REST servers in Python; JAX-RS is Java, AXIS is a SOAP stack.' },
    { q: 'Which statement comparing REST and SOAP is correct today?',
      o: ['REST services are more popular than SOAP services', 'The Header is a required part of a SOAP message', 'Browsers process XML more easily than JSON', 'UDDI is widely used in modern web services'], c: 0,
      r: 'REST/JSON has overtaken SOAP for most APIs; the SOAP Header is optional, browsers favour JSON, and UDDI is largely disused.' },
  ]},

  // ===== 22 · Cryptography & Signatures =====
  SEC: { title: '22 · Testownik — Cryptography & Signatures', q: [
    { q: 'Symmetric ciphers with a secret key:',
      o: ['Generally produce ciphertext the same size as the plaintext', 'Are no longer used because key storage is too cumbersome', 'Require keys of at least 1024 bits', 'Use a different key to encrypt and to decrypt'], c: 0,
      r: 'Symmetric ciphers reuse one secret key and output ciphertext roughly the size of the input (unlike public-key ciphers).' },
    { q: 'Encrypting a data stream with a block cipher in ECB (Electronic Code Book) mode means:',
      o: ['Each block is encrypted independently with the same cipher, key and IV', 'Each block is XORed with the previous ciphertext block (that is CBC)', 'Each block is XORed with a pseudorandom keystream (that is a stream cipher)', 'Each block uses an incrementing counter (that is CTR)'], c: 0,
      r: 'ECB encrypts every block independently with the same key/IV, so identical plaintext blocks give identical ciphertext (its weakness).' },
    { q: 'A cryptographic MAC (Message Authentication Code) is created by:',
      o: ['Encrypting a cryptographic hash of the message with a symmetric cipher / session key', 'Computing a plain CRC32 checksum of the payload', 'Encrypting the hash with the recipient’s public key', 'Sending an MD5 digest in clear text'], c: 0,
      r: 'A MAC is a message hash symmetrically encrypted with a shared session key (e.g. SHA1/RC4), giving integrity AND authenticity.' },
    { q: 'A digital (electronic) signature is created by:',
      o: ['Encrypting the message’s cryptographic hash with the signer’s private key', 'Encrypting the hash with the signer’s public key', 'Encrypting the hash with a symmetric cipher', 'Attaching the signer’s public certificate to the document'], c: 0,
      r: 'A signature = the document’s hash encrypted with the signer’s private key; anyone verifies it with the signer’s public key.' },
    { q: 'To verify data protected with a digital signature, you need:',
      o: ['The signer’s public key', 'The signer’s private key', 'The recipient’s private key', 'A shared symmetric secret'], c: 0,
      r: 'Verification decrypts the signature with the signer’s public key and compares hashes.' },
    { q: 'Securing transactions with a digital signature relies on:',
      o: ['Asymmetric (public-key) cryptography', 'Symmetric encryption', 'Hashing alone', 'Data compression'], c: 0,
      r: 'Digital signatures are an asymmetric-cryptography mechanism (private key signs, public key verifies).' },
    { q: 'Enveloping a message (a “digital envelope”) consists of encrypting it with:',
      o: ['A symmetric cipher under a random session key, with that key then encrypted using the recipient’s public key', 'A public cipher under the recipient’s public key applied to the whole message', 'A symmetric cipher under a pre-shared secret key', 'A symmetric cipher under a random key encrypted with the sender’s own private key'], c: 0,
      r: 'A digital envelope is hybrid: fast symmetric encryption of the data, then the random key is protected with the recipient’s public key.' },
    { q: 'A cryptographic hash function:',
      o: ['Maps data of any size to a fixed-size value and is one-way (not reversible)', 'Is a reversible two-way transform', 'Produces the same output for different inputs by design', 'Encrypts data with a shared secret key'], c: 0,
      r: 'A hash maps arbitrary input to a fixed-length digest; it is one-way and collision-resistant.' },
    { q: 'A hash-function collision is:',
      o: ['Two different messages m1 ≠ m2 with H(m1) = H(m2)', 'Two messages with different hash values', 'Two messages that merely have equal length', 'Two identical messages'], c: 0,
      r: 'A collision is two distinct inputs sharing the same hash output — the property good hash functions resist.' },
    { q: 'Which hash function is most vulnerable to collisions?',
      o: ['MD5', 'SHA-256', 'SHA-3', 'RIPEMD-160'], c: 0,
      r: 'MD5 is broken — practical collisions exist — so it must not be used for security.' },
  ]},

  // ===== 23 · Certificates & PKI =====
  PKI: { title: '23 · Testownik — Certificates & PKI', q: [
    { q: 'The Policy of Trust for public certificates:',
      o: ['Defines the principles for assessing the credibility of a certificate and the chain of CAs that signed it', 'Describes when an SSL/TLS connection is secure', 'Sets rules for using an employer-owned electronic signature', 'Tells an employee how to handle a digitally signed letter'], c: 0,
      r: 'The Policy of Trust governs how the trustworthiness of a certificate and its certification chain is judged (hierarchical PKIX, web-of-trust, etc.).' },
    { q: 'Which information MUST appear in a public certificate (per the RFC recommendations)?',
      o: ['The certificate’s serial number', 'A description of the certificate’s purpose', 'The owner’s contact email address', 'Subject Alternative Name (SAN) entries'], c: 0,
      r: 'Mandatory X.509 fields include serial number, validity period, issuer/subject DN and public key; purpose, email and SAN are optional.' },
    { q: 'An SSL/HTTPS server certificate must satisfy which additional requirement?',
      o: ['Its CN and/or SAN must match the server’s DNS name (and it must be signed by a trusted CA)', 'Its DN must match the server’s IP address', 'It must embed the server’s private key', 'It must declare the purpose it is used for'], c: 0,
      r: 'Per X.509/RFC 6125, hostname verification matches the DNS name against the certificate’s CN/SAN — never the DN against an IP address.' },
    { q: 'By issuing a PKIX certificate, the Certificate Authority (CA) guarantees that:',
      o: ['The public key in the certificate belongs to the entity named in the Distinguished Name (DN)', 'The certificate embeds the entity’s private key', 'The certificate is legally binding for banking transactions', 'Every optional field is filled in truthfully'], c: 0,
      r: 'A CA binds an identity (DN) to a public key; it vouches that the key belongs to the named entity.' },
    { q: 'The Certification Practice Statement specifies:',
      o: ['The procedures the CA uses to verify a certificate owner’s identity', 'How to tell whether an SSL connection is secure', 'How to check that a certificate has not expired', 'The list of revoked certificates'], c: 0,
      r: 'The CPS documents the CA’s operational procedures, including how it validates the identity of applicants.' },
    { q: 'The CRL (Certificate Revocation List) contains:',
      o: ['The serial numbers of certificates revoked by the CA', 'The numbers of certificates that have merely expired', 'An unprotected plain-text list anyone may edit', 'Certificates pushed instantly to every user'], c: 0,
      r: 'A CRL is a CA-signed list of revoked (not just expired) certificate serial numbers.' },
    { q: 'A distributed (web-of-trust) model of building trust in certificates/public keys:',
      o: ['Builds trust by finding mutual friends/acquaintances between the key owner and the recipient', 'Requires the certificate to be on a single central trusted list', 'Has never been implemented in practice', 'Requires maintaining huge central server farms'], c: 0,
      r: 'The distributed/OpenPGP model (web of trust) chains trust through mutual acquaintances — unlike hierarchical PKIX with a central Root CA.' },
    { q: 'When a browser connects to a server over HTTPS:',
      o: ['The client/user can be authenticated to the server with an X.509 client certificate', 'Server port 80 must be used', 'The user must be in the same DNS domain as the server', 'The server must be listed in a central database of trusted servers'], c: 0,
      r: 'HTTPS uses TLS; beyond authenticating the server, it can optionally authenticate the client with an X.509 certificate (mutual TLS).' },
  ]},

  // ===== 24 · Payments & E-commerce =====
  PAY: { title: '24 · Testownik — Payments & E-commerce', q: [
    { q: 'A model where consumers offer products/services and companies pay for them is abbreviated:',
      o: ['C2B', 'B2C', 'C2C', 'B2B'], c: 0,
      r: 'Consumer-to-Business (C2B): consumers set the offer/price and businesses buy.' },
    { q: 'Conducting online auctions is which type of relationship?',
      o: ['C2C', 'B2C', 'C2B', 'B2B'], c: 0,
      r: 'Auction marketplaces are mainly Consumer-to-Consumer (C2C).' },
    { q: 'The 3D Secure service increases payment security by:',
      o: ['Opening an additional connection to the card issuer’s server where the payer must authenticate', 'Putting the owner’s 3D biometric data in the card chip', 'Adding a 3D hologram to the payment card', 'Sending a 3D scan of the payer to the bank'], c: 0,
      r: '3D Secure adds an authentication step directly with the issuing bank, used for Card-Not-Present transactions.' },
    { q: 'The CVC2/CVV2/CID codes on credit cards are used:',
      o: ['To authorize “Card Not Present” transactions, e.g. online shopping', 'As a checksum verifying the card number', 'As a replacement PIN at ATMs', 'To set the ATM daily withdrawal limit'], c: 0,
      r: 'These codes prove the card is on hand for Card-Not-Present (online/phone) purchases.' },
    { q: 'The EMV protocol (Europay, MasterCard, Visa) is used for:',
      o: ['Cryptographic communication between the payment card and the POS terminal', 'Logging the cardholder into their online bank', 'Settling ATM withdrawals between operators', 'Sending verification tokens over the internet'], c: 0,
      r: 'EMV is the chip-card standard for secure card↔POS-terminal communication.' },
    { q: 'Card payments at a POS (Point-of-Sale) terminal:',
      o: ['Identify the card by reading the magnetic stripe or by exchanging data over EMV, depending on the card', 'Always identify the card by the number on the CVC1/CVV1 magnetic stripe', 'Are always registered on the account immediately', 'Are the safest transfer-based payment method'], c: 0,
      r: 'A POS terminal reads either the magnetic stripe or the EMV chip depending on the card’s capability.' },
    { q: 'Chargeback (the mechanism for reversing a bank transaction):',
      o: ['Can be used in the case of an unauthorized credit-card payment', 'Is possible for cash withdrawn from an ATM', 'Must be ordered by the payee (acquirer)', 'Applies to online bank-transfer payments'], c: 0,
      r: 'Chargeback lets a cardholder reverse an unauthorized or disputed card payment; it does not cover ATM cash or plain transfers.' },
    { q: 'The “electronic shopping experience” rules for online payment do NOT regulate:',
      o: ['Calculating the amount due based on the contents of the basket', 'Protection against withdrawing a payment after acceptance', 'Protection against identity theft by the store', 'How payment is confirmed with the bank'], c: 0,
      r: 'Cart pricing is store business logic; the payment rules cover acceptance, anti-fraud and bank confirmation, not how the total is computed.' },
  ]},

  // ===== 25 · Finance: WSE & Pensions =====
  FIN: { title: '25 · Testownik — Finance: WSE & Pensions', q: [
    { q: 'Fundamental analysis on the stock exchange is based on:',
      o: ['A company’s published financial reports', 'The share’s historical price chart', 'The NBP reference rate', 'The order/transaction history'], c: 0,
      r: 'Fundamental analysis values a company from its financial/economic reports; technical analysis instead studies the price history.' },
    { q: 'Technical analysis on the WSE uses:',
      o: ['The share’s historical price (chart) data', 'Quarterly financial reports', 'The NBP reference rate', 'The company’s board composition'], c: 0,
      r: 'Technical analysis predicts from past price/volume patterns, not company fundamentals.' },
    { q: 'The stock-exchange trading algorithm is used to:',
      o: ['Match the buy and sell orders submitted by investors', 'Compute the exchange index from transaction prices', 'Audit the correctness of executed transactions', 'Set the market’s reference interest rate'], c: 0,
      r: 'The matching engine pairs compatible buy/sell orders to execute trades.' },
    { q: 'Shares on the WSE in paper form:',
      o: ['Do not occur — shares are fully dematerialized', 'Occur normally', 'May occur under certain conditions', 'Occur only for state-owned companies'], c: 0,
      r: 'WSE shares are fully dematerialized (electronic records); paper certificates are not used.' },
    { q: 'The second pillar of the current Polish pension system is:',
      o: ['Voluntary, private, defined-contribution, funded', 'Compulsory, public, defined-contribution, funded', 'Compulsory, private, defined-benefit, funded', 'Voluntary, public, defined-benefit, pay-as-you-go'], c: 0,
      r: 'After the reforms, the 2nd pillar (OFE) is voluntary, privately managed, defined-contribution and funded.' },
    { q: 'A pay-as-you-go pension system (ZUS) is one in which:',
      o: ['Current contributions are used to pay the benefits currently being drawn', 'Contributions accrue in interest-bearing personal accounts', 'Contributions are revalued each year only by inflation', 'Contributions are invested on the stock market'], c: 0,
      r: 'Pay-as-you-go (repartition) funds today’s pensions directly from today’s contributions.' },
    { q: 'KSI ZUS is a system that is:',
      o: ['Centralized', 'Distributed', 'Hybrid', 'Peer-to-peer'], c: 0,
      r: 'KSI is the centralized central IT system of ZUS.' },
    { q: 'The approximate number of KSI ZUS payers (contributors) is:',
      o: ['About 5 million', 'About 1 million', 'About 18 million', 'About 39 million'], c: 0,
      r: 'The order of magnitude of ZUS contributors handled by KSI is a few million (~5M).' },
    { q: 'The first WSE trading system was created by:',
      o: ['Hewlett-Packard (HP)', 'Asseco', 'IBM', 'NYSE Technologies'], c: 0,
      r: 'The WSE’s first electronic trading system was built by HP; the current UTP system came later from NYSE Technologies.' },
    { q: 'The WSE’s currently used trading system (UTP) was created by:',
      o: ['NYSE Technologies', 'IBM', 'HP', 'Asseco'], c: 0,
      r: 'The Universal Trading Platform (UTP) in use today was supplied by NYSE Technologies.' },
    { q: 'The creator of the Edukacja.CL and JSOS 2.0 systems is:',
      o: ['Sygnity', 'Asseco', 'Comarch', 'IBM Poland'], c: 0,
      r: 'These academic systems were delivered by Sygnity.' },
  ]},

  // ===== 26 · Blockchain & Crypto =====
  BC: { title: '26 · Testownik — Blockchain & Crypto', q: [
    { q: 'What is a “nonce” in blockchain?',
      o: ['A “number used once” that miners vary to find a valid block hash', 'A QR code', 'A wallet address', 'A private key'], c: 0,
      r: 'In proof-of-work, the nonce is the field miners change to make the block hash meet the difficulty target.' },
    { q: 'The genesis block is:',
      o: ['The first block in the chain', 'The most recently mined block', 'A block created dynamically that holds the previous hash', 'A block that has no hash'], c: 0,
      r: 'The genesis block is the hard-coded first block; it has no predecessor.' },
    { q: 'The term “block hash” means:',
      o: ['A digest (hash) of all the data/transactions in the block', 'A wallet address', 'A transaction’s QR code', 'An encrypted private key'], c: 0,
      r: 'The block hash is the cryptographic digest of the block header (which commits to all its transactions).' },
    { q: 'The components of the “equation” in a Proof-of-Work block are:',
      o: ['Previous block hash, Merkle-tree root, nonce, timestamp', 'Previous block hash, a single transaction hash, nonce, timestamp', 'Previous block hash, transaction hash, nonce, Schnorr signature', 'Only the nonce and the timestamp'], c: 0,
      r: 'A Bitcoin-style header hashes the previous block hash, the Merkle root (committing all transactions), a nonce and a timestamp.' },
    { q: 'The most common consensus mechanism in blockchain is:',
      o: ['Proof of Work', 'Proof of Stake', 'Proof of Concept', 'Proof of Authority'], c: 0,
      r: 'Proof of Work (mining) is the original and still most widespread consensus mechanism (e.g. Bitcoin).' },
    { q: 'The BTC difficulty value is:',
      o: ['Adjusted periodically based on how fast recent blocks were added', 'Fixed forever at the genesis block', 'Changed exactly every two weeks regardless of block times', 'Set manually by individual miners'], c: 0,
      r: 'Difficulty retargets so the average block time stays roughly constant as total hash power changes — it sets the block-production rate.' },
    { q: 'What is an ICO?',
      o: ['A method of raising funds by issuing new tokens (Initial Coin Offering)', 'A transaction-encryption protocol', 'A type of cryptocurrency wallet', 'A user identity-verification method'], c: 0,
      r: 'An ICO raises capital by selling newly issued crypto tokens to investors.' },
    { q: 'A so-called hard fork is:',
      o: ['An incompatible protocol change that splits the chain in two', 'A temporary, backward-compatible update', 'An attack on the peer-to-peer network', 'A change of a wallet’s private key'], c: 0,
      r: 'A hard fork is a non-backward-compatible rule change; nodes that do not upgrade follow a separate chain.' },
    { q: 'A smart contract is:',
      o: ['A program stored on the blockchain that automatically executes the contract’s terms', 'A type of cryptocurrency wallet', 'A physical document signed digitally', 'A transaction-encryption algorithm'], c: 0,
      r: 'A smart contract is self-executing code on-chain that enforces agreed terms without an intermediary.' },
    { q: 'Which feature of blockchain makes stored data impossible to change?',
      o: ['Immutability (hash-linked blocks)', 'Anonymity of the database', 'Absence of cryptography', 'Absence of digital signatures'], c: 0,
      r: 'Each block commits to the previous block’s hash, so altering past data would invalidate every later block — that is immutability.' },
  ]},

  // ===== 27 · Cloud, Containers & DevOps =====
  CLOUD: { title: '27 · Testownik — Cloud, Containers & DevOps', q: [
    { q: 'What does the command `kubectl get pods` do?',
      o: ['Lists the active pods in the current namespace', 'Creates a new pod in the cluster', 'Restarts all pods in the cluster', 'Deletes all pods in the namespace'], c: 0,
      r: '`kubectl get pods` lists the pods running in the current namespace.' },
    { q: 'Which command starts the services defined in docker-compose.yml?',
      o: ['docker-compose up', 'docker-compose build', 'docker run compose', 'docker compose start'], c: 0,
      r: '`docker-compose up` builds (if needed) and starts all services defined in the file.' },
    { q: 'Which file does Docker Compose use by default to configure services?',
      o: ['docker-compose.yml', 'docker-compose.json', 'Dockerfile.yml', 'compose.config'], c: 0,
      r: 'Docker Compose reads docker-compose.yml (or compose.yaml) by default.' },
    { q: 'What is the name of the file that defines a CI/CD pipeline in GitLab?',
      o: ['.gitlab-ci.yml', '.ci.yml', '.gitlab-pipeline.yml', 'pipeline.yaml'], c: 0,
      r: 'GitLab CI/CD reads the pipeline definition from .gitlab-ci.yml in the repo root.' },
    { q: 'What does a Deployment do in Kubernetes?',
      o: ['Manages the scaling and rolling updates of a set of Pods', 'Creates secret data for containers', 'Monitors network traffic between Pods', 'Configures the load balancer automatically'], c: 0,
      r: 'A Deployment declaratively manages a ReplicaSet of Pods, handling scaling and rolling updates.' },
    { q: 'What is a Pod in Kubernetes?',
      o: ['The smallest deployable unit, holding one or more containers', 'A replica of a cluster managed by kube-apiserver', 'The networking layer between services', 'A group of clusters managed together'], c: 0,
      r: 'A Pod is the smallest deployable K8s unit; it wraps one or more tightly-coupled containers sharing network/storage.' },
    { q: 'Which Kubernetes object defines rules for routing incoming (external) traffic to services?',
      o: ['Ingress', 'ConfigMap', 'Node', 'Pod'], c: 0,
      r: 'An Ingress defines host/path rules that route external HTTP(S) traffic to in-cluster services.' },
    { q: 'What is the difference between a Docker container and a Docker image?',
      o: ['The image is the built template; the container is its running instance', 'The image is a running instance of a container', 'They are the same thing in Docker', 'The container is the template; the image is the runtime'], c: 0,
      r: 'An image is an immutable built template; running it produces a container (its live instance).' },
    { q: 'Which set contains only type-1 (bare-metal) hypervisors?',
      o: ['Xen and VMware vSphere', 'VMware Workstation and VirtualBox', 'VirtualBox and Xen', 'VMware Workstation and vSphere'], c: 0,
      r: 'Type-1 (bare-metal) hypervisors run directly on hardware (Xen, vSphere); type-2 (hosted) run on a host OS (VirtualBox, VMware Workstation).' },
    { q: 'Paravirtualization was proposed by the company:',
      o: ['Xen', 'VMware', 'Docker', 'Microsoft'], c: 0,
      r: 'The Xen project introduced paravirtualization, where the guest OS is modified to cooperate with the hypervisor.' },
    { q: 'In which cloud service model does the customer manage ONLY the applications and data?',
      o: ['PaaS', 'SaaS', 'IaaS', 'On-premises'], c: 0,
      r: 'In PaaS the customer manages only application + data; the provider runs the runtime, OS and below. (In SaaS the provider manages those too; in IaaS the customer also manages the OS.)' },
    { q: 'What is the role of an exchange in RabbitMQ?',
      o: ['It receives messages from producers and routes them to the appropriate queues', 'It stores durable messages in a queue', 'It processes messages and returns replies directly', 'It monitors the queue and drops inactive messages'], c: 0,
      r: 'A RabbitMQ exchange takes producer messages and, by its bindings/routing keys, places them into the right queues.' },
    { q: 'Load balancers working at the network/transport layer keep a user’s session on one server by:',
      o: ['Routing requests based on the client’s IP address', 'Analyzing cookie contents', 'Relying on DNS round-robin', 'Using an SSL accelerator'], c: 0,
      r: 'Layer-3/4 balancers have no view of cookies, so they pin sessions by source IP (DNS round-robin balances poorly).' },
  ]},

  // ===== 28 · ITIL & Project Management =====
  ITSM: { title: '28 · Testownik — ITIL & Project Management', q: [
    { q: 'ITIL is:',
      o: ['A set of best-practice guidelines and a common language for IT and business', 'A rigid set of mandatory rules', 'Equivalent to ISO 27001', 'A software product you install'], c: 0,
      r: 'ITIL is a framework of best-practice guidance (not a law or product) that aligns IT with business.' },
    { q: 'What defines the value of a service according to ITIL?',
      o: ['The perceived benefits and usefulness of the service to the customer', 'The cost of implementing the service', 'The number of users of the service', 'The uptime achieved without failure'], c: 0,
      r: 'Service value is the customer’s perceived benefit/usefulness — the combination of utility and warranty.' },
    { q: 'Which statement describes the utility of an IT service?',
      o: ['The functionality that meets the customer’s need — “fit for purpose”', 'The guarantee of availability — “fit for use”', 'The incident response time', 'The cost of implementing the service'], c: 0,
      r: 'Utility = what the service does (fit for purpose); warranty = how well it performs (fit for use).' },
    { q: 'Which term describes the warranty (guarantee) of a service?',
      o: ['“Fit for use” — the assurance it will perform as agreed', '“Fit for purpose”', 'Cost efficiency', 'Operational novelty'], c: 0,
      r: 'Warranty is the assurance the service meets agreed levels (availability, capacity, security, continuity) — fit for use.' },
    { q: 'Which element is one of the four dimensions of service management in ITIL 4?',
      o: ['Partners and suppliers', 'SLA tools', 'KPI metrics', 'Audit reports'], c: 0,
      r: 'The four dimensions: organizations & people; information & technology; partners & suppliers; value streams & processes.' },
    { q: 'In a Service Desk context, SPOC stands for:',
      o: ['Single Point of Contact', 'Standard Procedure Operational Control', 'System Performance Oversight Committee', 'Service Process Optimization Center'], c: 0,
      r: 'SPOC = Single Point of Contact — the one place users interact with the service provider.' },
    { q: 'Which of the following is NOT a goal of incident management?',
      o: ['Removing the root cause of incidents', 'Restoring normal service as quickly as possible', 'Logging and categorizing tickets', 'Keeping users informed of incident status'], c: 0,
      r: 'Incident management restores service ASAP; eliminating the underlying root cause is Problem Management.' },
    { q: 'Which activity does NOT belong to the Service Operation phase?',
      o: ['Designing new services', 'Fulfilling user requests', 'Monitoring performance', 'Logging incidents'], c: 0,
      r: 'Designing new services is Service Design; Service Operation runs and supports live services.' },
    { q: 'Which ITIL role is responsible for approving the budget for IT services?',
      o: ['Sponsor', 'User', 'Supplier', 'Service-desk agent'], c: 0,
      r: 'The Sponsor authorizes the funding/budget for the service.' },
    { q: 'A Partnership / Multi-sourcing strategy means:',
      o: ['Sharing service provision over the lifecycle with two or more organizations', 'Outsourcing specific processes (e.g. HR, payroll)', 'Using external resources for one defined area (e.g. cleaners)', 'Running on-demand apps on shared external computers'], c: 0,
      r: 'Multi-sourcing/partnership shares delivery responsibility across the lifecycle among several organizations.' },
    { q: 'Which statement about a project is true?',
      o: ['It is planned, rigorously executed and frequently controlled', 'It consists of repetitive, ongoing phases', 'It delivers minimum quality whenever convenient', 'It never needs monitoring once started'], c: 0,
      r: 'A project is a temporary, planned endeavour that is executed and controlled against defined parameters.' },
    { q: 'The classic success criteria (“triple constraint”) of a project are:',
      o: ['Budget, time and scope', 'Time, resources and people', 'Features, resources and schedule', 'Only the absence of delays'], c: 0,
      r: 'The triple constraint balances scope, time (schedule) and cost (budget).' },
    { q: 'With CPI = 0.24 and SPI = 1.01, the correct conclusion is:',
      o: ['None of the listed conclusions is correct', 'The project is behind schedule but on budget', 'The cost to complete is below plan and all is well', 'No corrective action is needed'], c: 0,
      r: 'CPI 0.24 ≪ 1 means a severe cost overrun; SPI ≈ 1 means on schedule. The other options misread these, so none is correct.' },
  ]},
};

// ---- assemble exam sets: shuffle each question's option order (de-bias position) ----
const ORDER = ['WS', 'SEC', 'PKI', 'PAY', 'FIN', 'BC', 'CLOUD', 'ITSM'];
const examSets = ORDER.map(key => {
  const def = EXAM[key];
  return {
    title: def.title, group: 'exam',
    questions: def.q.map(item => {
      const order = shuffle(item.o.map((_, i) => i));
      const options = order.map(i => item.o[i]);
      const correct = order.indexOf(item.c);
      if (options.length !== 4 || correct < 0 || correct > 3) {
        console.error('BAD exam Q:', item.q);
      }
      return { question: item.q, options, correct, reasoning: item.r };
    }),
  };
});

// ---- combine + assign global ids ----
const allSets = [...studySets, ...examSets];
let gid = 0, setId = 0;
allSets.forEach(s => {
  s.id = ++setId;
  s.questions.forEach(q => { q.id = ++gid; });
});

// ---- emit chapters.js ----
function esc(s) { return JSON.stringify(s); }
let out = `// Question bank for "IT Applications in Business and Commerce".
// GENERATED by _build_bank.js — DO NOT hand-edit. Edit chapters.backup.js (study) or the
// EXAM array in _build_bank.js (Testownik), then re-run: node _build_bank.js
//
// Two groups (see set.group): "study" = concept questions (option order de-biased so the
// answer is not guessable by position); "exam" = real past-exam concepts, de-duplicated
// (the raw bank repeated many items) and each answer verified against uploads/ITABC.md.
//
// Shape: quizData.sets[] = { id, title, group, questions[] }
//   question = { id, question, options:[4], correct:<0-3>, reasoning }
const quizData = {
    sets: [
`;
allSets.forEach((s, si) => {
  out += `        { id: ${s.id}, title: ${esc(s.title)}, group: ${esc(s.group)}, questions: [\n`;
  s.questions.forEach(q => {
    out += `            {id:${q.id},question:${esc(q.question)},options:[${q.options.map(esc).join(',')}],correct:${q.correct},reasoning:${esc(q.reasoning)}},\n`;
  });
  out += `        ]}${si < allSets.length - 1 ? ',' : ''}\n`;
});
out += `    ]\n};\n`;
fs.writeFileSync('chapters.js', out);

const totalQ = allSets.reduce((n, s) => n + s.questions.length, 0);
const examQ = examSets.reduce((n, s) => n + s.questions.length, 0);
console.log('Wrote chapters.js:', allSets.length, 'sets,', totalQ, 'questions (study:', studySets.length, 'sets; exam:', examSets.length, 'sets /', examQ, 'Q)');
examSets.forEach(s => console.log('  ', s.title, '->', s.questions.length));
