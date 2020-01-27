package submarine

import (
	"bufio"
	"errors"
	"github.com/golang/glog"
	"math/rand"
	"os"
	"strings"
	"time"
)

const (
	defaultClientTimeout = 2 * time.Second
	defaultClientName    = ""

	// ErrNotFound cannot find a node to connect to
	ErrNotFound = "Unable to find a node to connect"
)

// AdminConnectionsInterface interface representing the map of admin connections to submarine cluster nodes
type AdminConnectionsInterface interface {
	// Add connect to the given address and
	// register the client connection to the pool
	Add(addr string) error
	// Remove disconnect and remove the client connection from the map
	Remove(addr string)
	// Get returns a client connection for the given address,
	// connects if the connection is not in the map yet
	Get(addr string) (ClientInterface, error)
	// GetRandom returns a client connection to a random node of the client map
	GetRandom() (ClientInterface, error)
	// GetDifferentFrom returns a random client connection different from given address
	GetDifferentFrom(addr string) (ClientInterface, error)
	// GetAll returns a map of all clients per address
	GetAll() map[string]ClientInterface
	//GetSelected returns a map of clients based on the input addresses
	GetSelected(addrs []string) map[string]ClientInterface
	// Reconnect force a reconnection on the given address
	// if the adress is not part of the map, act like Add
	Reconnect(addr string) error
	// AddAll connect to the given list of addresses and
	// register them in the map
	// fail silently
	AddAll(addrs []string)
	// ReplaceAll clear the map and re-populate it with new connections
	// fail silently
	ReplaceAll(addrs []string)
	// ValidateResp check the submarine resp, eventually reconnect on connection error
	// in case of error, customize the error, log it and return it
	///ValidateResp(resp *submarine.Resp, addr, errMessage string) error
	// ValidatePipeResp wait for all answers in the pipe and validate the response
	// in case of network issue clear the pipe and return
	// in case of error return false
	// ValidatePipeResp(c ClientInterface, addr, errMessage string) bool
	// Reset close all connections and clear the connection map
	Reset()
}

// AdminConnections connection map for submarine cluster
// currently the admin connection is not threadSafe since it is only use in the Events thread.
type AdminConnections struct {
	clients           map[string]ClientInterface
	connectionTimeout time.Duration
	commandsMapping   map[string]string
	clientName        string
}

// Add connect to the given address and
// register the client connection to the map
func (cnx *AdminConnections) Add(addr string) error {
	_, err := cnx.Update(addr)
	return err
}

// Update returns a client connection for the given adress,
// connects if the connection is not in the map yet
func (cnx *AdminConnections) Update(addr string) (ClientInterface, error) {
	// if already exist close the current connection
	if c, ok := cnx.clients[addr]; ok {
		c.Close()
	}

	c, err := cnx.connect(addr)
	if err == nil && c != nil {
		cnx.clients[addr] = c
	} else {
		glog.V(3).Infof("Cannot connect to %s ", addr)
	}
	return c, err
}

func (cnx *AdminConnections) connect(addr string) (ClientInterface, error) {
	c, err := NewClient(addr, cnx.connectionTimeout, cnx.commandsMapping)
	if err != nil {
		return nil, err
	}
	if cnx.clientName != "" {
		///resp := c.Cmd("CLIENT", "SETNAME", cnx.clientName)
		///return c, cnx.ValidateResp(resp, addr, "Unable to run command CLIENT SETNAME")
	}

	return c, nil
}

// AddAll connect to the given list of addresses and
// register them in the map
// fail silently
func (cnx *AdminConnections) AddAll(addrs []string) {
	for _, addr := range addrs {
		cnx.Add(addr)
	}
}

// buildCommandReplaceMapping reads the config file with the command-replace lines and build a mapping of
// bad lines are ignored silently
func buildCommandReplaceMapping(filePath string) map[string]string {
	mapping := make(map[string]string)
	file, err := os.Open(filePath)
	if err != nil {
		glog.Errorf("Cannor open %s: %v", filePath, err)
		return mapping
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		elems := strings.Fields(scanner.Text())
		if len(elems) == 3 && strings.ToLower(elems[0]) == "rename-command" {
			mapping[strings.ToUpper(elems[1])] = elems[2]
		}
	}

	if err := scanner.Err(); err != nil {
		glog.Errorf("Cannor parse %s: %v", filePath, err)
		return mapping
	}
	return mapping
}

// NewAdminConnections returns and instance of AdminConnectionsInterface
func NewAdminConnections(addrs []string, options *AdminOptions) AdminConnectionsInterface {
	cnx := &AdminConnections{
		clients:           make(map[string]ClientInterface),
		connectionTimeout: defaultClientTimeout,
		commandsMapping:   make(map[string]string),
		clientName:        defaultClientName,
	}
	if options != nil {
		if options.ConnectionTimeout != 0 {
			cnx.connectionTimeout = options.ConnectionTimeout
		}
		if _, err := os.Stat(options.RenameCommandsFile); err == nil {
			cnx.commandsMapping = buildCommandReplaceMapping(options.RenameCommandsFile)
		}
		cnx.clientName = options.ClientName
	}
	cnx.AddAll(addrs)
	return cnx
}

// ReplaceAll clear the pool and re-populate it with new connections
// fail silently
func (cnx *AdminConnections) ReplaceAll(addrs []string) {
	cnx.Reset()
	cnx.AddAll(addrs)
}

// Reset close all connections and clear the connection map
func (cnx *AdminConnections) Reset() {
	for _, c := range cnx.clients {
		c.Close()
	}
	cnx.clients = map[string]ClientInterface{}
}

// Remove disconnect and remove the client connection from the map
func (cnx *AdminConnections) Remove(addr string) {
	if c, ok := cnx.clients[addr]; ok {
		c.Close()
		delete(cnx.clients, addr)
	}
}

// Get returns a client connection for the given adress,
// connects if the connection is not in the map yet
func (cnx *AdminConnections) Get(addr string) (ClientInterface, error) {
	if c, ok := cnx.clients[addr]; ok {
		return c, nil
	}
	c, err := cnx.connect(addr)
	if err == nil && c != nil {
		cnx.clients[addr] = c
	}
	return c, err
}

// GetRandom returns a client connection to a random node of the client map
func (cnx *AdminConnections) GetRandom() (ClientInterface, error) {
	_, c, err := cnx.getRandomKeyClient()
	return c, err
}

// GetRandom returns a client connection to a random node of the client map
func (cnx *AdminConnections) getRandomKeyClient() (string, ClientInterface, error) {
	nbClient := len(cnx.clients)
	if nbClient == 0 {
		return "", nil, errors.New(ErrNotFound)
	}
	randNumber := rand.Intn(nbClient)
	for k, c := range cnx.clients {
		if randNumber == 0 {
			return k, c, nil
		}
		randNumber--
	}

	return "", nil, errors.New(ErrNotFound)
}

// GetDifferentFrom returns random a client connection different from given address
func (cnx *AdminConnections) GetDifferentFrom(addr string) (ClientInterface, error) {
	if len(cnx.clients) == 1 {
		for a, c := range cnx.clients {
			if a != addr {
				return c, nil
			}
			return nil, errors.New(ErrNotFound)
		}
	}

	for {
		a, c, err := cnx.getRandomKeyClient()
		if err != nil {
			return nil, err
		}
		if a != addr {
			return c, nil
		}
	}
}

// GetAll returns a map of all clients per address
func (cnx *AdminConnections) GetAll() map[string]ClientInterface {
	return cnx.clients
}

//GetSelected returns a map of clients based on the input addresses
func (cnx *AdminConnections) GetSelected(addrs []string) map[string]ClientInterface {
	clientsSelected := make(map[string]ClientInterface)
	for _, addr := range addrs {
		if client, ok := cnx.clients[addr]; ok {
			clientsSelected[addr] = client
		}
	}
	return clientsSelected
}

// Reconnect force a reconnection on the given address
// is the adress is not part of the map, act like Add
func (cnx *AdminConnections) Reconnect(addr string) error {
	glog.Infof("Reconnecting to %s", addr)
	cnx.Remove(addr)
	return cnx.Add(addr)
}
