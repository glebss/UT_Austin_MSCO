//!
//! coordinator.rs
//! Implementation of 2PC coordinator
//!
extern crate log;
extern crate stderrlog;
extern crate rand;
extern crate ipc_channel;

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::channel;
use std::thread;
use std::time::Duration;
use std::io;

use coordinator::ipc_channel::ipc::IpcSender as Sender;
use coordinator::ipc_channel::ipc::IpcReceiver as Receiver;
use coordinator::ipc_channel::ipc::TryRecvError;
// use coordinator::ipc_channel::ipc::channel;

use message;
use message::MessageType;
use message::ProtocolMessage;
use message::RequestStatus;
use oplog;
use crate::participant::Participant;
use crate::client::Client;

/// CoordinatorState
/// States for 2PC state machine
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoordinatorState {
    Quiescent,
    ReceivedRequest,
    ProposalSent,
    ReceivedVotesAbort,
    ReceivedVotesCommit,
    SentGlobalDecision
}

/// Coordinator
/// Struct maintaining state for coordinator
#[derive(Debug)]
pub struct Coordinator<'a> {
    id_str: String,
    state: CoordinatorState,
    running: Arc<AtomicBool>,
    log: oplog::OpLog,
    participants: Vec<&'a Participant<'a>>,
    clients: Vec<&'a Client<'a>>,
    paticipants_senders: Vec<Sender<ProtocolMessage>>,
    participants_receivers: Vec<Receiver<ProtocolMessage>>,
    clients_senders: Vec<Sender<ProtocolMessage>>,
    clients_receivers: Vec<Receiver<ProtocolMessage>>
}

///
/// Coordinator
/// Implementation of coordinator functionality
/// Required:
/// 1. new -- Constructor
/// 2. protocol -- Implementation of coordinator side of protocol
/// 3. report_status -- Report of aggregate commit/abort/unknown stats on exit.
/// 4. participant_join -- What to do when a participant joins
/// 5. client_join -- What to do when a client joins
///
impl<'a> Coordinator<'a> {

    ///
    /// new()
    /// Initialize a new coordinator
    ///
    /// <params>
    ///     log_path: directory for log files --> create a new log there.
    ///     r: atomic bool --> still running?
    ///
    pub fn new(
        id_str: String,
        log_path: String,
        r: &Arc<AtomicBool>,
        sender: Sender<ProtocolMessage>,
        receiver: Receiver<ProtocolMessage>) -> Coordinator {

        Coordinator {
            id_str: id_str.clone(),
            state: CoordinatorState::Quiescent,
            log: oplog::OpLog::new(log_path),
            running: r.clone(),
            participants: Vec::new(),
            clients: Vec::new(),
            paticipants_senders: Vec::new(),
            participants_receivers: Vec::new(),
            clients_senders: Vec::new(),
            clients_receivers: Vec::new()
        }
    }

    ///
    /// participant_join()
    /// Adds a new participant for the coordinator to keep track of
    ///
    /// HINT: Keep track of any channels involved!
    /// HINT: You may need to change the signature of this function
    ///
    pub fn participant_join(&mut self, name: &'a String, mut participant: Participant) -> Result<(), io::Error> {
        assert!(self.state == CoordinatorState::Quiescent);
        let (sender_participant, receiver_coordinator) = ipc_channel::ipc::channel().unwrap();
        let (sender_coordinator, receiver_participant) = ipc_channel::ipc::channel().unwrap();
        participant.sender = sender_participant;
        participant.receiver = receiver_participant;
        self.paticipants_senders.push(sender_coordinator);
        self.participants_receivers.push(receiver_coordinator);
        Ok(())
        // TODO
    }

    ///
    /// client_join()
    /// Adds a new client for the coordinator to keep track of
    ///
    /// HINT: Keep track of any channels involved!
    /// HINT: You may need to change the signature of this function
    ///
    pub fn client_join(&mut self, name: &'a String, mut client: Client) -> Result<(), io::Error> {
        assert!(self.state == CoordinatorState::Quiescent);
        let (sender_client, receiver_coordinator) = ipc_channel::ipc::channel().unwrap();
        let (sender_coordinator, receiver_client) = ipc_channel::ipc::channel().unwrap();
        client.sender = sender_client;
        client.receiver = receiver_client;
        self.clients_senders.push(sender_coordinator);
        self.clients_receivers.push(receiver_coordinator);
        Ok(())
        // TODO
    }

    ///
    /// report_status()
    /// Report the abort/commit/unknown status (aggregate) of all transaction
    /// requests made by this coordinator before exiting.
    ///
    pub fn report_status(&mut self) {
        // TODO: Collect actual stats
        let successful_ops: u64 = 0;
        let failed_ops: u64 = 0;
        let unknown_ops: u64 = 0;

        println!("coordinator     :\tCommitted: {:6}\tAborted: {:6}\tUnknown: {:6}", successful_ops, failed_ops, unknown_ops);
    }

    ///
    /// protocol()
    /// Implements the coordinator side of the 2PC protocol
    /// HINT: If the simulation ends early, don't keep handling requests!
    /// HINT: Wait for some kind of exit signal before returning from the protocol!
    ///
    pub fn protocol(&mut self, txid: &'a String) {

        // let msg_from_client = self.receiver.try_recv();
        // let mut txid: String = "".to_string();
        let mut opid: u32 = 0;
        // match msg_from_client {
        //     Ok(msg) => {
        //         txid = msg.txid.clone();
        //         opid = msg.opid;
        //     }
        //     Err(err) => {}
        // }
        self.state = CoordinatorState::ReceivedRequest;
        // phase 1
        for (idx, p) in self.participants.iter().enumerate() {
            opid += 1;
            let pm = message::ProtocolMessage::generate(message::MessageType::CoordinatorPropose,
                txid.clone(),
                self.id_str.clone(),
                opid);
            let _ = self.paticipants_senders[idx].send(pm);
        }
        self.state = CoordinatorState::ProposalSent;
        

        self.report_status();
    }
}
