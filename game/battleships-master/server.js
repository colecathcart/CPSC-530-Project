//CPSC 530 Group 3 final project
//=======================================
//This is an edited version of an online
//battleship game that was NOT created by
//the members of group 3. The original version
//is published here: https://github.com/kubowania/battleships
//The original game is published under a public MIT license
//with full consent from the original authors to reuse and edit
//this software. The original license is included in the README.md file
//in this directory

const express = require('express')
const path = require('path')
const http = require('http')
const fs = require('fs')
const PORT = process.env.PORT || 3000
const socketio = require('socket.io')
const app = express()
const server = http.createServer(app)
const io = socketio(server)

let moves = ""

// Set static folder
app.use(express.static(path.join(__dirname, "public")))

// Start server
server.listen(PORT, () => console.log(`Server running on port ${PORT}`))

// Handle a socket connection request from web client
const connections = [null, null]

io.on('connection', socket => {
  // console.log('New WS Connection')

  // Find an available player number
  let playerIndex = -1;
  for (const i in connections) {
    if (connections[i] === null) {
      playerIndex = i
      break
    }
  }

  // Tell the connecting client what player number they are
  socket.emit('player-number', playerIndex)

  console.log(`Player ${playerIndex} has connected`)

  // Ignore player 3
  if (playerIndex === -1) return

  connections[playerIndex] = false

  // Tell eveyone what player number just connected
  socket.broadcast.emit('player-connection', playerIndex)

  // Handle Diconnect
  socket.on('disconnect', () => {
    console.log(`Player ${playerIndex} disconnected`)
    connections[playerIndex] = null
    //Tell everyone what player numbe just disconnected
    socket.broadcast.emit('player-connection', playerIndex)
  })

  // On Ready
  socket.on('player-ready', () => {
    socket.broadcast.emit('enemy-ready', playerIndex)
    connections[playerIndex] = true
  })

  // Check player connections
  socket.on('check-players', () => {
    const players = []
    for (const i in connections) {
      connections[i] === null ? players.push({connected: false, ready: false}) : players.push({connected: true, ready: connections[i]})
    }
    socket.emit('check-players', players)
  })

  // On Fire Received
  socket.on('fire', id => {
    console.log(`Shot fired from ${playerIndex}`, id)

    // Emit the move to the other player
    socket.broadcast.emit('fire', id)
  })
  // on Fire Reply
  socket.on('fire-reply', square => {
    console.log(square)
    const obj = Object.values(square)
    if (obj.includes('miss') && !obj.includes('inprogress')){
      console.log('Record move!')
      moves = moves.concat(obj[1])
      fs.writeFile(__dirname + '/public/moves.txt', moves, (err) => {
          
        // In case of a error throw err.
        if (err){
          console.log(err)
        };
    })
    }
    // Forward the reply to the other player
    socket.broadcast.emit('fire-reply', square)
  })

  // Timeout connection
  setTimeout(() => {
    connections[playerIndex] = null
    socket.emit('timeout')
    socket.disconnect()
  }, 600000) // 10 minute limit per player
})